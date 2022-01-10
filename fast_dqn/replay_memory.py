import numpy as np


class ReplayMemory:
    def __init__(self, env, capacity, seed):
        self._env = env
        self._nominal_capacity = capacity
        self._history_len = None
        self._size_now = 0
        self._pointer = 0
        self._np_random = np.random.RandomState(seed)

        self._allocated = False

    def _allocate(self, observations):
        # Divide the nominal capacity by the number of environments
        self._num_envs = observations.shape[0]
        assert (self._nominal_capacity % self._num_envs) == 0
        self._capacity = self._nominal_capacity // self._num_envs

        # Allocate memory for the buffers
        self.observations = np.empty([self._capacity, *observations.shape],
                                        dtype=observations.dtype)
        self.actions = np.empty([self._capacity, self._num_envs], dtype=self._env.action_space.dtype)
        self.rewards = np.empty([self._capacity, self._num_envs], dtype=np.float32)
        self.dones = np.empty([self._capacity, self._num_envs], dtype=np.bool)

    def save(self, states, actions, rewards, dones):
        # NOTE: Assumes observations are stacked on last axis to form states
        observations = states[..., -1, None]

        if not self._allocated:
            self._allocate(observations)
            self._allocated = True
            self._history_len = states.shape[-1]

        p = self._pointer
        self.observations[p], self.actions[p], self.rewards[p], self.dones[p] = observations, actions, rewards, dones
        self._size_now = min(self._size_now + 1, self._capacity)
        self._pointer = (p + 1) % self._capacity

    def sample(self, batch_size):
        indices = self._np_random.randint(self._size_now - 1, size=batch_size)
        x = self._absolute_location(indices)
        states = self._get_states(indices)
        next_states = self._get_states(indices + 1)
        vec_minibatch = tuple((states, self.actions[x], self.rewards[x], next_states, self.dones[x]))

        # So far, we have a minibatch of `num_env * batch_size` samples.
        # We need to randomly sample from axis 1 to get down to `batch_size` samples.
        rand_ints = self._np_random.randint(self._num_envs, size=batch_size)
        return map(lambda x: x[np.arange(batch_size), rand_ints], vec_minibatch)

    def _get_states(self, indices):
        # NOTE: Indices must be relative to the pointer
        states = []
        for j in reversed(range(self._history_len)):
            # Let i be the index relative to the pointer and let x be the absolute location
            i = indices - j
            x = self._absolute_location(i)

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

    def _absolute_location(self, relative_index):
        return (self._pointer + relative_index) % self._size_now
