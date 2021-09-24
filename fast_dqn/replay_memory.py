import numpy as np

from fast_dqn.image_stacker import ImageStacker


class ReplayMemory:
    def __init__(self, env, capacity, history_len, seed):
        self._capacity = capacity
        self._history_len = history_len
        self._size_now = 0
        self._pointer = 0
        self._np_random = np.random.RandomState(seed)

        # Warning: Assumes observations are stacked on last axis to form state
        obs_shape = (*env.observation_space.shape[:-1], 1)
        self._image_stacker = ImageStacker(history_len)

        self.observations = np.empty(shape=[capacity, *obs_shape],
                                     dtype=env.observation_space.dtype)
        self.actions = np.empty(shape=[capacity], dtype=env.action_space.dtype)
        self.rewards = np.empty(capacity, dtype=np.float32)
        self.dones = np.empty(capacity, dtype=np.bool)

    def save(self, observation, action, reward, done):
        p = self._pointer
        self.observations[p], self.actions[p], self.rewards[p], self.dones[p] = observation, action, reward, done
        self._size_now = min(self._size_now + 1, self._capacity)
        self._pointer = (p + 1) % self._capacity

    def sample(self, batch_size):
        indices = self._np_random.randint(self._size_now - 1, size=batch_size)
        x = self._absolute(indices)

        states = np.stack([self._get_state(i) for i in indices])
        next_states = np.stack([self._get_state(i + 1) for i in indices])

        return (states, self.actions[x], self.rewards[x], next_states, self.dones[x])

    def _get_state(self, index):
        self._image_stacker._reset()

        for i in range(index - self._history_len + 1, index + 1):
            if i == 0:
                # This is the oldest experience; zero out all previous frames
                self._image_stacker._reset()

            # Add the current frame to the stack
            x = self._absolute(i)
            self._image_stacker.append(self.observations[x])

            if i == index:
                # The image stack is full so we are done
                return self._image_stacker.get_stack()

            if self.dones[x]:
                # This episode is done; zero out all previous frames
                self._image_stacker._reset()

    def _absolute(self, i):
        return (self._pointer + i) % self._size_now
