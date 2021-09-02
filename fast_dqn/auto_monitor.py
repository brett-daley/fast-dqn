from threading import Thread
import time
from queue import Queue

import gym
import numpy as np


class _GlobalMonitor:
    def __init__(self):
        # These metrics are never reset:
        self._episodes = 0
        self._steps = 0

        # Metric histories for averaging
        self._all_lengths = []
        self._all_returns = []

        self._task_queue = Queue()
        Thread(target=self._task_loop, daemon=True).start()

        # Print the header
        self._print('episode', 'timestep', 'length', 'return', 'avg_length',
                    'avg_return', 'hours', sep=',', flush=True)

        # Initial time reference point
        self._start_time = time.time()

    def episode_done(self, episode_length, episode_return):
        self._task_queue.put_nowait((episode_length, episode_return))

    def _task_loop(self):
        while True:
            episode_length, episode_return = self._task_queue.get()
            assert isinstance(episode_length, int)
            assert isinstance(episode_return, float)

            self._episodes += 1
            self._steps += episode_length

            self._all_lengths.append(episode_length)
            self._all_returns.append(episode_return)

            hours = (time.time() - self._start_time) / 3600
            avg_length = np.mean(self._all_lengths[-100:])
            avg_return = np.mean(self._all_returns[-100:])

            self._print(self._episodes, self._steps, episode_length, episode_return, avg_length,
                        avg_return, '{:.3f}'.format(hours), sep=',', flush=True)

            self._task_queue.task_done()

    def _print(self, *args, **kwargs):
        print("AM", end=':')
        print(*args, **kwargs)


class AutoMonitor(gym.Wrapper):
    global_monitor = _GlobalMonitor()

    def __init__(self, env):
        super().__init__(env)

        # These metrics are reset when an episode ends:
        self._length = None
        self._return = None

    def step(self, action):
        observation, reward, done, info = super().step(action)
        self._length += 1
        self._return += reward
        if done:
            AutoMonitor.global_monitor.episode_done(self._length, self._return)
        return observation, reward, done, info

    def reset(self, **kwargs):
        self._length = 0
        self._return = 0.0
        return super().reset(**kwargs)
