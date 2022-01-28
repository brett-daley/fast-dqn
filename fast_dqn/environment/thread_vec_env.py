from operator import itemgetter
from threading import Thread
from queue import Queue

import gym
import numpy as np


class ThreadVecEnv:
    def __init__(self, env_fns):
        self._shmem = shmem = SharedMemory()
        self._envs = tuple(EnvWorker(fn(), i, shmem) for i, fn in enumerate(env_fns))
        self.num_envs = len(self._envs)
        self.observation_space = self._envs[0].observation_space
        self.action_space = self._envs[0].action_space

    def step(self, actions):
        self.step_async(actions)
        return self.step_wait()

    def step_async(self, actions):
        assert len(actions) == self.num_envs
        for env, action in zip(self._envs, actions):
            env.step_async(action)

    def step_wait(self):
        observations, rewards, dones = self._envs[0].step_wait()
        return observations, rewards, dones, None

    def reset(self):
        for env in self._envs:
            env.reset()
        obs, _, _ = self._shmem.read()
        return obs

    def seed(self, seed=None):
        for env in self._envs:
            env.seed(seed)

    def close(self):
        for env in self._envs:
            env.close()


class EnvWorker(gym.Wrapper):
    def __init__(self, env, worker_id, shared_memory):
        self.shmem = shared_memory
        super().__init__(env)
        self._id = worker_id
        self._action_queue = Queue()
        # self._result_queue = Queue()
        Thread(target=self._action_loop, daemon=True).start()

    def _action_loop(self):
        while True:
            a = self._action_queue.get()
            observation, reward, done, info = self.env.step(a)
            self.shmem.write(self._id, (observation, reward, done))
            # self._result_queue.put_nowait(result)
            self._action_queue.task_done()

    def step_async(self, action):
        self._action_queue.put_nowait(action)

    def step_wait(self):
        self._action_queue.join()
        return self.shmem.read()


class SharedMemory:
    def __init__(self):
        N = 8
        self.observations = np.empty([N, *(84, 84, 4)], dtype=np.uint8)
        self.rewards = np.empty(N, dtype=np.float32)
        self.dones = np.empty(N, dtype=np.bool)

    def read(self):
        return (self.observations.copy(), self.rewards.copy(), self.dones.copy())

    def write(self, worker_id, transition):
        observation, reward, done = transition
        self.observations[worker_id] = observation
        self.rewards[worker_id] = reward
        self.dones[worker_id] = done
