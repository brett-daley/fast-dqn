import time

import gym
import numpy as np
from stable_baselines3.common.vec_env import SubprocVecEnv


class VecMonitor:
    def __init__(self, vec_env):
        self._enabled = True
        self._silent = False

        assert isinstance(vec_env, SubprocVecEnv)
        self.env = vec_env
        self.num_envs = self.env.num_envs
        self.remotes = self.env.remotes

        # These metrics are never reset:
        self._episodes = 0
        self._steps = 0

        # Metric histories for averaging
        self._all_lengths = []
        self._all_returns = []

    @property
    def observation_space(self):
        return self.env.observation_space

    @property
    def action_space(self):
        return self.env.action_space

    def step(self, actions):
        observations, rewards, dones, infos = self.env.step(actions)

        if self._enabled:
            if any(dones):
                self._log_done_episodes()

        return observations, rewards, dones, infos

    def reset(self):
        if self._steps == 0:
            # Print the header
            self._print('episode', 'timestep', 'length', 'return', 'avg_length',
                'avg_return', 'hours', sep=',', flush=True)
            # Initial time reference point
            self._start_time = time.time()
        return self.env.reset()

    def seed(self, seed=None):
        return self.env.seed(seed)

    def _log_done_episodes(self):
        for r in self.remotes:
            r.send(('flush', None))
        results = [r.recv() for r in self.remotes]

        for episode in results:
            if episode is not None:
                length, ret = episode
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
        if self._silent:
            return
        print("AM", end=':')
        print(*args, **kwargs)

    def enable_monitor(self, enable):
        assert isinstance(enable, bool)
        self._enabled = enable

    def silence_monitor(self, silence):
        assert isinstance(silence, bool)
        self._silent = silence

    def get_episode_returns(self):
        return list(self._all_returns)


class Monitor(gym.Wrapper):
    def __init__(self, env):
        super().__init__(env)

        # These metrics are reset when an episode ends:
        self._length = None
        self._return = None

        # Tracks the length/return for the most recently completed episode
        self._last_episode = None

    def step(self, action):
        observation, reward, done, info = self.env.step(action)
        self._length += 1
        self._return += reward
        if done:
            self._last_episode = (self._length, self._return)
        return observation, reward, done, info

    def reset(self, **kwargs):
        self._length = 0
        self._return = 0.0
        return super().reset(**kwargs)

    def flush(self):
        if self._last_episode is not None:
            episode = tuple(self._last_episode)
        else:
            episode = None
        self._last_episode = None
        return episode
