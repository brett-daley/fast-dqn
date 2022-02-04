import itertools
from operator import itemgetter
import multiprocessing
from threading import Thread
from queue import Queue

import gym
import numpy as np

from fast_dqn.environment.replay_memory import ReplayMemory


class ThreadVecEnv:
    def __init__(self, env_fns, rmem_fn):
        self.num_envs = len(env_fns)
        context = multiprocessing.get_context()

        self.pipes, worker_pipes = zip(*[context.Pipe() for _ in range(self.num_envs)])
        self.processes = []
        for pipe, env_fn in zip(worker_pipes, env_fns):
            args = (pipe, env_fn, rmem_fn)
            proc = context.Process(target=env_worker, args=args, daemon=True)
            proc.start()
            self.processes.append(proc)

        self.pipes[0].send(('get_spaces', None))
        self.observation_space, self.action_space = self.pipes[0].recv()

    def step(self, actions):
        self.step_async(actions)
        return self.step_wait()

    def step_async(self, actions):
        assert len(actions) == self.num_envs
        for pipe, action in zip(self.pipes, actions):
            pipe.send(('step', action))

    def step_wait(self):
        results = [pipe.recv() for pipe in self.pipes]
        states, rewards, dones, infos = zip(*results)
        states, rewards, dones = map(np.stack, (states, rewards, dones))
        return states, rewards, dones, infos

    def reset(self):
        [pipe.send(('reset', None)) for pipe in self.pipes]
        states = [pipe.recv() for pipe in self.pipes]
        return np.stack(states)

    def seed(self, seed):
        for i, pipe in enumerate(self.pipes):
            pipe.send(('seed', seed + i))

    def close(self):
        for pipe in self.pipes:
            pipe.send(('close', None))

    def sample_replay_memory(self, batch_size):
        assert (batch_size % self.num_envs) == 0
        per_env_batch_size = batch_size // self.num_envs
        [pipe.send(('sample_replay_memory', per_env_batch_size)) for pipe in self.pipes]

        minibatches = [pipe.recv() for pipe in self.pipes]
        states, actions, rewards, next_states, dones = map(
            lambda x: list(itertools.chain.from_iterable(x)), zip(*minibatches))
        return map(np.stack, (states, actions, rewards, next_states, dones))

    def get_last_episode(self, env_id):
        self.pipes[env_id].send(('get_last_episode', None))
        return self.pipes[env_id].recv()


def env_worker(pipe, env_fn, rmem_fn):
    env = env_fn()
    replay_memory = rmem_fn()
    state = None

    def step(data):
        global state
        action = data
        next_state, reward, done, info = env.step(action)
        replay_memory.save(state, action, reward, done)
        if done:
            next_state = env.reset()
        pipe.send((next_state, reward, done, info))

    def reset(data):
        assert data is None
        global state
        state = env.reset()
        pipe.send(state)

    def seed(data):
        seed = data
        env.seed(seed)

    def close(data):
        assert data is None
        env.close()
        pipe.close()

    def sample_replay_memory(data):
        batch_size = data
        minibatch = replay_memory.sample(batch_size)
        pipe.send(minibatch)

    def get_spaces(data):
        assert data is None
        pipe.send((env.observation_space, env.action_space))

    def get_last_episode(data):
        assert data is None
        pipe.send(env.last_episode)

    while True:
        try:
            cmd, data = pipe.recv()
            locals()[cmd](data)
            if cmd == 'close':
                break
        except EOFError:
            break
