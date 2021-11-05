from threading import Thread
from queue import Queue

import numpy as np

from run_dqn import DQNAgent


class Worker:
    def __init__(self, worker_id, env, agent):
        self._id = worker_id
        self._env = env
        self._agent = agent

        self._state = env.reset()

        self._transition_buffer = []
        self._completed_episodes = []

        self._sample_queue = Queue()
        Thread(target=self._sample_loop, daemon=True).start()

    def update(self, t):
        self._sample_queue.put_nowait(t)

    def _sample_loop(self):
        while True:
            t = self._sample_queue.get()
            epsilon = DQNAgent.epsilon_schedule(t)
            self._step(epsilon)
            self._sample_queue.task_done()

    def policy(self, state, epsilon):
        assert 0.0 <= epsilon <= 1.0
        # With probability epsilon, take a random action
        if self._env.action_space.np_random.rand() <= epsilon:
            return self._env.action_space.sample()

        # Otherwise, compute the greedy (i.e. best predicted) action
        return np.argmax(self._get_qvalues(state))

    @property
    def _state(self):
        return self._agent.shared_states[self._id]

    @_state.setter
    def _state(self, state):
        self._agent.shared_states[self._id] = state

    def _get_qvalues(self, state):
        if self._agent._synchronize:
            # The agent has pre-computed Q-values for us
            return self._agent.shared_qvalues[self._id]
        else:
            # We use the target network here so we can train the main network in parallel
            return self._agent._dqn.predict_target(state[None])[0]

    def _step(self, epsilon):
        action = self.policy(self._state, epsilon)
        next_state, reward, done, info = self._env.step(action)

        self._transition_buffer.append( (self._state.copy(), action, reward, done) )

        if done:
            self._completed_episodes.append(self._transition_buffer)
            self._transition_buffer = []
            self._state = self._env.reset()
        else:
            self._state = next_state

        return next_state, reward, done, info

    def join(self):
        self._sample_queue.join()

    def flush(self):
        for episode in self._completed_episodes:
            for transition in episode:
                yield transition
        self._completed_episodes.clear()
