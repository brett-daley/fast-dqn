import itertools
from operator import itemgetter
from threading import Thread
from queue import Queue

import gym
import numpy as np

from fast_dqn.environment.replay_memory import ReplayMemory


class ThreadVecEnv:
    def __init__(self, env_fns, rmem_capacity):
        self.num_envs = len(env_fns)
        self._envs = tuple(EnvWorker(fn()) for fn in env_fns)
        self.observation_space = self._envs[0].observation_space
        self.action_space = self._envs[0].action_space

        self.has_replay_memory = False
        self._rmem_capacity = rmem_capacity

    def allocate_replay_memory(self):
        assert not self.has_replay_memory
        assert (self._rmem_capacity % self.num_envs) == 0
        per_env_capacity = self._rmem_capacity // self.num_envs

        for i, env in enumerate(self._envs):
            env.replay_memory = ReplayMemory(self.action_space, capacity=per_env_capacity)
            env.replay_memory.seed(self._seed + i)

        self.has_replay_memory = True

    def step(self, actions):
        self.step_async(actions)
        return self.step_wait()

    def step_async(self, actions):
        assert len(actions) == self.num_envs
        for env, action in zip(self._envs, actions):
            env.step_async(action)

    def step_wait(self):
        results = [env.step_wait() for env in self._envs]
        states, rewards, dones, infos = zip(*results)
        states, rewards, dones = map(np.stack, (states, rewards, dones))
        return states, rewards, dones, infos

    def reset(self):
        states = [env.reset() for env in self._envs]
        return np.stack(states)

    def seed(self, seed):
        self._seed = seed
        for i, env in enumerate(self._envs):
            env.seed(seed + i)

    def close(self):
        for env in self._envs:
            env.close()

    def sample_replay_memory(self, batch_size):
        assert self.has_replay_memory
        assert (batch_size % self.num_envs) == 0
        per_env_batch_size = batch_size // self.num_envs
        minibatches = [env.replay_memory.sample(per_env_batch_size) for env in self._envs]
        states, actions, rewards, next_states, dones = map(
            lambda x: list(itertools.chain.from_iterable(x)), zip(*minibatches))
        return map(np.stack, (states, actions, rewards, next_states, dones))


class EnvWorker(gym.Wrapper):
    def __init__(self, env):
        super().__init__(env)
        self._action_queue = Queue()
        Thread(target=self._action_loop, daemon=True).start()
        self.replay_memory = None

    def reset(self):
        self._state = super().reset()
        return self._state.copy()

    def _action_loop(self):
        while True:
            action = self._action_queue.get()

            self._last_result = next_state, reward, done, info = self.env.step(action)
            self.replay_memory.save(self._state, action, reward, done)
            if done:
                next_state = self.reset()
            self._state = next_state

            self._action_queue.task_done()

    def step_async(self, action):
        self._action_queue.put_nowait(action)

    def step_wait(self):
        self._action_queue.join()
        return self._last_result
