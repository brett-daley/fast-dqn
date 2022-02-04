import time

import gym
import numpy as np

from fast_dqn.environment.thread_vec_env import ThreadVecEnv


class VecMonitor:
    def __init__(self, vec_env):
        assert isinstance(vec_env, ThreadVecEnv)
        self.env = vec_env

        # These metrics are never reset:
        self._episodes = 0
        self._steps = 0

        # Metric histories for averaging
        self._all_lengths = []
        self._all_returns = []

    def __getattr__(self, name):
        return getattr(self.env, name)

    def step(self, actions):
        observations, rewards, dones, infos = self.env.step(actions)
        for i, done in enumerate(dones):
            if done:
                self._log_episode(*self.env.get_last_episode(env_id=i))
        return observations, rewards, dones, infos

    def reset(self):
        if self._steps == 0:
            # Print the header
            self._print('episode', 'timestep', 'length', 'return', 'avg_length',
                'avg_return', 'hours', sep=',', flush=True)
            # Initial time reference point
            self._start_time = time.time()
        return self.env.reset()

    def _log_episode(self, length, ret):
        assert isinstance(length, int)
        assert isinstance(ret, float)
        self._episodes += 1
        self._steps += length

        self._all_lengths.append(length)
        self._all_returns.append(ret)

        hours = (time.time() - self._start_time) / 3600
        avg_length = np.mean(self._all_lengths[-100:])
        avg_return = np.mean(self._all_returns[-100:])

        self._print(self._episodes, self._steps, length, ret, avg_length,
                    avg_return, '{:.3f}'.format(hours), sep=',', flush=True)

    def _print(self, *args, **kwargs):
        print("AM", end=':')
        print(*args, **kwargs)

    def get_episode_returns(self):
        return list(self._all_returns)


class Monitor(gym.Wrapper):
    def __init__(self, env):
        super().__init__(env)

        # These metrics are reset when an episode ends:
        self._length = None
        self._return = None

        # Tracks the length/return for the most recently completed episode
        self.last_episode = None

    def step(self, action):
        observation, reward, done, info = self.env.step(action)
        self._length += 1
        self._return += reward
        if done:
            self.last_episode = (self._length, self._return)
        return observation, reward, done, info

    def reset(self, **kwargs):
        self._length = 0
        self._return = 0.0
        return super().reset(**kwargs)
