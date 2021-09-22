import numpy as np


class ReplayMemory:
    def __init__(self, env, capacity, history_len, seed):
        self._capacity = capacity
        self._history_len = history_len
        self._size_now = 0
        self._pointer = 0
        self._np_random = np.random.RandomState(seed)

        # Warning: Assumes observations are stacked on last axis to form state
        obs_shape = (*env.observation_space.shape[:-1], 1)

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
        j = self._np_random.randint(self._size_now - 1, size=batch_size)
        j = (self._pointer + j) % self._size_now
        return (self._get_states(j),
                self.actions[j],
                self.rewards[j],
                self._get_states((j + 1) % self._size_now),
                self.dones[j])

    def _get_states(self, indices):
        states = []
        for j in reversed(range(self._history_len)):
            x = (indices - j) % self._size_now
            states.append(self.observations[x])

        mask = np.ones_like(states[0])
        for j in range(1, self._history_len):
            i = indices - j
            x = i % self._size_now
            mask[self.dones[x]] = 0.0
            mask[np.where(i < 0)] = 0.0
            states[-1 - j] *= mask

        states = np.concatenate(states, axis=-1)
        assert states.shape[0] == len(indices)
        assert (states.shape[-1] % self._history_len) == 0
        return states
