from operator import itemgetter
from threading import Thread
from queue import Queue

import gym
import numpy as np


class ThreadVecEnv:
    def __init__(self, env_fns):
        self._envs = tuple(EnvWorker(fn()) for fn in env_fns)
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
        obs = []
        r = []
        d = []
        for env in self._envs:
            observation, reward, done, _ = env.step_wait()
            if done:
                observation = env.reset()
            obs.append(observation)
            r.append(reward)
            d.append(done)
        return np.stack(obs), np.stack(r), np.stack(d), None

    def reset(self):
        observations = []
        for env in self._envs:
            obs = env.reset()
            observations.append(obs)
        return np.stack(observations)

    def seed(self, seed=None):
        for env in self._envs:
            env.seed(seed)

    def close(self):
        for env in self._envs:
            env.close()


class EnvWorker(gym.Wrapper):
    def __init__(self, env):
        super().__init__(env)
        self._action_queue = Queue()
        Thread(target=self._action_loop, daemon=True).start()

    def _action_loop(self):
        while True:
            a = self._action_queue.get()
            self._last_result = self.env.step(a)
            self._action_queue.task_done()

    def step_async(self, action):
        self._action_queue.put_nowait(action)

    def step_wait(self):
        self._action_queue.join()
        return self._last_result
