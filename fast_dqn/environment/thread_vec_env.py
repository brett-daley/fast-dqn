import multiprocessing

import numpy as np


class ThreadVecEnv:
    def __init__(self, env_fns, replay_memory):
        self.rmem = replay_memory
        self.num_envs = len(env_fns)
        context = multiprocessing.get_context()

        self.pipes, worker_pipes = zip(*[context.Pipe() for _ in range(self.num_envs)])
        self.processes = []
        for pipe, env_fn in zip(worker_pipes, env_fns):
            proc = context.Process(target=env_worker, args=(pipe, env_fn), daemon=True)
            proc.start()
            self.processes.append(proc)

        self.pipes[0].send(('get_spaces', None))
        self.observation_space, self.action_space = self.pipes[0].recv()

        # Array containing the current state of each env instance
        self._states = None

    def step(self, actions):
        assert len(actions) == self.num_envs
        for pipe, action in zip(self.pipes, actions):
            pipe.send(('step', action))
        results = [pipe.recv() for pipe in self.pipes]
        next_states, rewards, dones, infos = zip(*results)
        self.rmem.save(self._states.copy(), actions, rewards, dones)
        self._states, rewards, dones = map(np.stack, (next_states, rewards, dones))
        return self._states, rewards, dones, infos

    def reset(self):
        [pipe.send(('reset', None)) for pipe in self.pipes]
        states = [pipe.recv() for pipe in self.pipes]
        self._states = np.stack(states)
        return self._states

    def seed(self, seed):
        for i, pipe in enumerate(self.pipes):
            pipe.send(('seed', seed + i))

    def close(self):
        for pipe in self.pipes:
            pipe.send(('close', None))

    def get_last_episode(self, env_id):
        self.pipes[env_id].send(('get_last_episode', None))
        return self.pipes[env_id].recv()


def env_worker(pipe, env_fn):
    env = env_fn()
    state = None

    def step(data):
        action = data
        next_state, reward, done, info = env.step(action)
        if done:
            next_state = env.reset()
        pipe.send((next_state, reward, done, info))

    def reset(data):
        assert data is None
        nonlocal state
        state = env.reset()
        pipe.send(state)

    def seed(data):
        seed = data
        env.seed(seed)

    def close(data):
        assert data is None
        env.close()
        pipe.close()

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
