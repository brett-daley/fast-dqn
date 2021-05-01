import time

import gym
import numpy as np


class AutoMonitor(gym.Wrapper):
    def __init__(self, env):
        super().__init__(env)

        # These metrics are never reset:
        self._episodes = 0
        self._steps = 0
        # These metrics are reset when an episode ends:
        self._length = None
        self._return = None

        # Metric histories for averaging
        self._all_lengths = []
        self._all_returns = []

        # Initial time reference point
        self._start_time = time.time()

        # Print the header
        self._print('episode', 'timestep', 'length', 'return', 'avg_length',
                    'avg_return', 'hours', sep=',', flush=True)

    def step(self, action):
        observation, reward, done, info = super().step(action)
        self._on_step(reward)
        if done:
            self._on_done()
        return observation, reward, done, info

    def reset(self, **kwargs):
        self._on_reset()
        return super().reset(**kwargs)

    def _on_step(self, reward):
        self._steps += 1
        self._length += 1
        self._return += reward

    def _on_done(self):
        self._episodes += 1
        self._all_lengths.append(self._length)
        self._all_returns.append(self._return)

        hours = (time.time() - self._start_time) / 3600
        avg_length = np.mean(self._all_lengths[-100:])
        avg_return = np.mean(self._all_returns[-100:])

        self._print(self._episodes, self._steps, self._length, self._return, avg_length,
                    avg_return, '{:.3f}'.format(hours), sep=',', flush=True)

    def _on_reset(self):
        self._length = 0
        self._return = 0.0

    def _print(self, *args, **kwargs):
        print("AM", end=':')
        print(*args, **kwargs)
