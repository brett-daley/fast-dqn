import numpy as np


class ReplayMemory:
    def __init__(self, env, capacity):
        self._capacity = capacity
        self._size_now = 0
        self._pointer = 0

        self.states = np.empty(shape=[capacity, *env.observation_space.shape],
                               dtype=env.observation_space.dtype)
        self.actions = np.empty(shape=[capacity], dtype=env.action_space.dtype)
        self.rewards = np.empty(capacity, dtype=np.float32)
        self.dones = np.empty(capacity, dtype=np.float32)

    def save(self, state, action, reward, done):
        p = self._pointer
        self.states[p], self.actions[p], self.rewards[p], self.dones[p] = state, action, reward, done
        self._size_now = min(self._size_now + 1, self._capacity)
        self._pointer = (p + 1) % self._capacity

    def sample(self, batch_size):
        j = np.random.randint(self._size_now - 1, size=batch_size)
        j = (self._pointer + j) % self._size_now
        return (self.states[j],
                self.actions[j],
                self.rewards[j],
                self.states[(j + 1) % self._size_now],
                self.dones[j])

    def insert_most_recent(self, minibatch, next_state):
        states, actions, rewards, next_states, dones = minibatch
        i = (self._pointer - 1) % self._size_now
        states[0] = self.states[i]
        actions[0] = self.actions[i]
        rewards[0] = self.rewards[i]
        next_states[0] = next_state
        dones[0] = self.dones[i]
        return (states, actions, rewards, next_states, dones)
