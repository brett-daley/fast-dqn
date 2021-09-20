from threading import Lock
import time
from collections import defaultdict

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

        # Temporary storage between flushes
        self._episode_buffer = defaultdict(list)
        self._lock = Lock()
        self._i = 0

        # Print the header
        self._print('episode', 'timestep', 'length', 'return', 'avg_length',
                    'avg_return', 'hours', sep=',', flush=True)

        # Initial time reference point
        self._start_time = time.time()

    def register_id(self):
        self._i += 1
        return int(self._i)

    def episode_done(self, id_number, episode_length, episode_return):
        time_done = time.time()
        with self._lock:
            # Organized by ID number to make ordering deterministic
            self._episode_buffer[id_number].append( (episode_length, episode_return, time_done) )

    def flush(self):
        with self._lock:
            for key in range(1, self._i + 1):
                for episode_length, episode_return, time_done in self._episode_buffer[key]:
                    assert isinstance(episode_length, int)
                    assert isinstance(episode_return, float)

                    self._episodes += 1
                    self._steps += episode_length

                    self._all_lengths.append(episode_length)
                    self._all_returns.append(episode_return)

                    hours = (time_done - self._start_time) / 3600
                    avg_length = np.mean(self._all_lengths[-100:])
                    avg_return = np.mean(self._all_returns[-100:])

                    self._print(self._episodes, self._steps, episode_length, episode_return, avg_length,
                                avg_return, '{:.3f}'.format(hours), sep=',', flush=True)

            self._episode_buffer.clear()

    def _print(self, *args, **kwargs):
        print("AM", end=':')
        print(*args, **kwargs)


class AutoMonitor(gym.Wrapper):
    global_monitor = _GlobalMonitor()

    def __init__(self, env):
        super().__init__(env)
        self._enabled = True
        self._auto_flush = False
        self._id = AutoMonitor.global_monitor.register_id()

        # These metrics are reset when an episode ends:
        self._length = None
        self._return = None

    def step(self, action):
        observation, reward, done, info = self.env.step(action)
        self._length += 1
        self._return += reward
        if done:
            if self._enabled:
                AutoMonitor.global_monitor.episode_done(self._id, self._length, self._return)
                if self._auto_flush:
                    self.flush_monitor()
        return observation, reward, done, info

    def reset(self, **kwargs):
        self._length = 0
        self._return = 0.0
        return super().reset(**kwargs)

    def enable_monitor(self, enable, auto_flush=None):
        self._enabled = enable
        if auto_flush is not None:
            # Warning: do not enable auto flush if running parallel environments
            self._auto_flush = auto_flush

    def flush_monitor(self):
        AutoMonitor.global_monitor.flush()
