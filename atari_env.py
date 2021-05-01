from collections import deque

import cv2
import gym
from gym.envs.atari.atari_env import AtariEnv
import numpy as np

from auto_monitor import AutoMonitor


def make(game, size=84, grayscale=True, history_len=4):
    env = AtariEnv(game, frameskip=4, obs_type='image')
    env = AutoMonitor(env)
    if 'FIRE' in env.unwrapped.get_action_meanings():
        env = FireResetWrapper(env)
    env = NoopResetWrapper(env)
    env = EpisodicLifeWrapper(env)
    env = ClippedRewardWrapper(env)
    env = PreprocessedImageWrapper(env, size, grayscale)
    if history_len > 1:
        env = HistoryWrapper(env, history_len)
    return env


class ClippedRewardWrapper(gym.RewardWrapper):
    """Clips rewards to be in {-1, 0, +1} based on their signs."""
    def reward(self, reward):
        return np.sign(reward)


class EpisodicLifeWrapper(gym.Wrapper):
    """Signals done when a life is lost, but only resets when the game ends."""
    def __init__(self, env):
        super().__init__(env)
        self.lives = 0
        self.was_real_done = True

    def step(self, action):
        self.observation, reward, done, info = self.env.step(action)
        self.was_real_done = done
        lives = self.env.unwrapped.ale.lives()
        if 0 < lives < self.lives:
            # We lost a life, but force a reset only if it's not game over.
            # Otherwise, the environment just handles it automatically.
            done = True
        self.lives = lives
        return self.observation, reward, done, info

    def reset(self):
        if self.was_real_done:
            self.observation = self.env.reset()
        self.lives = self.env.unwrapped.ale.lives()
        return self.observation


class FireResetWrapper(gym.Wrapper):
    """Take action on reset for environments that are fixed until firing."""
    def __init__(self, env):
        assert env.unwrapped.get_action_meanings()[1] == 'FIRE'
        super().__init__(env)

    def reset(self):
        self.env.reset()
        observation, _, _, _ = self.step(1)
        return observation


class HistoryWrapper(gym.Wrapper):
    """Stacks the previous `history_len` observations along their last axis.
    Pads observations with zeros at the beginning of an episode."""
    def __init__(self, env, history_len=4):
        assert history_len > 1
        super().__init__(env)
        self.history_len = history_len
        self.deque = deque(maxlen=history_len)

        self.shape = self.observation_space.shape
        self.dtype = self.observation_space.dtype
        self.observation_space.shape = (*self.shape[:-1], history_len * self.shape[-1])

    def step(self, action):
        observation, reward, done, info = self.env.step(action)
        self.deque.append(observation)
        return self._history(), reward, done, info

    def reset(self):
        observation = self.env.reset()
        self._clear()
        self.deque.append(observation)
        return self._history()

    def _history(self):
        return np.concatenate(list(self.deque), axis=-1)

    def _clear(self):
        for _ in range(self.history_len):
            self.deque.append(np.zeros(self.shape, dtype=self.dtype))


class NoopResetWrapper(gym.Wrapper):
    """Sample initial states by taking a random number of no-ops on reset.
    The number is sampled uniformly from [0, `noop_max`]."""
    def __init__(self, env, noop_max=30):
        assert env.unwrapped.get_action_meanings()[0] == 'NOOP'
        super().__init__(env)
        self.noop_max = noop_max

    def reset(self):
        observation = self.env.reset()
        n = np.random.randint(self.noop_max + 1)
        for _ in range(n):
            observation, _, _, _ = self.step(0)
        return observation


class PreprocessedImageWrapper(gym.ObservationWrapper):
    """Resizes image observations and optionally converts them to grayscale."""
    def __init__(self, env, size=84, grayscale=True):
        super().__init__(env)
        self.size = size
        self.grayscale = grayscale
        self.shape = (size, size, 1 if grayscale else 3)
        self.observation_space = gym.spaces.Box(low=0, high=255, shape=self.shape, dtype=np.uint8)

    def observation(self, observation):
        if self.grayscale:
            observation = cv2.cvtColor(observation, cv2.COLOR_BGR2GRAY)
        observation = cv2.resize(observation, (self.size, self.size), interpolation=cv2.INTER_LINEAR)
        return observation.reshape(self.shape).astype(np.uint8)
