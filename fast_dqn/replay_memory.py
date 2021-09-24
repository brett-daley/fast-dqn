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

    def save(self, state, action, reward, done):
        observation = state[..., -1, None]
        p = self._pointer
        self.observations[p], self.actions[p], self.rewards[p], self.dones[p] = observation, action, reward, done
        self._size_now = min(self._size_now + 1, self._capacity)
        self._pointer = (p + 1) % self._capacity

    def sample(self, batch_size):
        indices = self._np_random.randint(self._size_now - 1, size=batch_size)
        x = self._absolute(indices)

        states = self._get_states(indices)
        next_states = self._get_states(indices + 1)

        return (states, self.actions[x], self.rewards[x], next_states, self.dones[x])

    def _get_states(self, indices):
        # NOTE: Indices are relative to the pointer
        states = []
        for j in reversed(range(self._history_len)):
            # Let i be the index relative to the pointer and let x be the absolute location
            i = indices - j
            x = self._absolute(i)

            # Zero out frames where we have wrapped around the replay memory
            for s in states:
                s[np.where(i == 0)] = 0.0

            # Add the current frame to the stack
            states.append(self.observations[x])

            if j == 0:
                # The image stack is full so we are done
                break

            # Zero out frames where the episode has terminated
            for s in states:
                s[self.dones[x]] = 0.0

        states = np.concatenate(states, axis=-1)
        assert states.shape[0] == len(indices)
        assert (states.shape[-1] % self._history_len) == 0
        return states

    def _absolute(self, i):
        return (self._pointer + i) % self._size_now
