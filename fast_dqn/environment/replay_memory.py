import itertools

import numpy as np


class ReplayMemory:
    def __init__(self, capacity, num_envs, seed):
        def rmem_fn():
            assert (capacity % num_envs) == 0
            rmem = ScalarReplayMemory(capacity // num_envs)
            rmem.seed(seed)
            return rmem

        self.num_envs = num_envs
        self.rmems = [rmem_fn() for _ in range(num_envs)]
        # Temporarily holds transitions until flush() is explicitly called
        self._transition_buffer = []

    def size(self):
        return sum([rmem._size_now for rmem in self.rmems])

    def save(self, states, actions, rewards, dones):
        transition = (states, actions, rewards, dones)
        self._transition_buffer.append(transition)

    def _write(self, states, actions, rewards, dones):
        for i, rmem in enumerate(self.rmems):
            rmem.save(states[i], actions[i], rewards[i], dones[i])

    def sample(self, batch_size):
        assert (batch_size % self.num_envs) == 0
        per_env_batch_size = batch_size // self.num_envs
        minibatches = [rmem.sample(per_env_batch_size) for rmem in self.rmems]
        states, actions, rewards, next_states, dones = map(
            lambda x: list(itertools.chain.from_iterable(x)), zip(*minibatches))
        return map(np.stack, (states, actions, rewards, next_states, dones))

    def flush(self):
        for transition in self._transition_buffer:
            self._write(*transition)
        self._transition_buffer.clear()


class ScalarReplayMemory:
    def __init__(self, capacity):
        self._capacity = capacity
        self._history_len = None
        self._size_now = 0
        self._pointer = 0
        self._np_random = None
        self._allocated = False

    def seed(self, seed):
        self._np_random = np.random.RandomState(seed)

    def _allocate(self, observation, action):
        action = np.array(action)
        # Allocate memory for the buffers
        self.observations = np.empty([self._capacity, *observation.shape],
                                     dtype=observation.dtype)
        self.actions = np.empty([self._capacity, *action.shape],
                                dtype=action.dtype)
        self.rewards = np.empty([self._capacity], dtype=np.float32)
        self.dones = np.empty([self._capacity], dtype=np.bool)
        self._allocated = True

    def save(self, state, action, reward, done):
        # NOTE: Assumes observations are stacked on last axis to form states
        observation = state[..., -1, None]

        if not self._allocated:
            self._allocate(observation, action)
            self._history_len = state.shape[-1]

        p = self._pointer
        self.observations[p], self.actions[p], self.rewards[p], self.dones[p] = observation, action, reward, done
        self._size_now = min(self._size_now + 1, self._capacity)
        self._pointer = (p + 1) % self._capacity

    def sample(self, batch_size):
        indices = self._np_random.randint(self._size_now - 1, size=batch_size)
        x = self._absolute_location(indices)
        states = self._get_states(indices)
        next_states = self._get_states(indices + 1)
        return (states, self.actions[x], self.rewards[x], next_states, self.dones[x])

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
