import itertools

import numpy as np

from fast_dqn.deep_q_network import DeepQNetwork
from fast_dqn.environment import VecMonitor


class BaselineDQNAgent:
    def __init__(self, make_vec_env_fn, instances, pytorch, **kwargs):
        self._make_vec_env_fn = make_vec_env_fn
        self._instances = instances

        self._vec_env = env = make_vec_env_fn(instances)
        self.action_space = self._vec_env.action_space

        self._dqn = DeepQNetwork(env, discount=0.99, pytorch=pytorch)

        self._prepopulate = 50_000
        self._train_freq = 4
        self._batch_size = 32
        self._target_update_freq = 10_000

    def run(self, duration):
        env = self._vec_env
        states = env.reset()
        assert (self._prepopulate % self._instances) == 0
        for _ in range(self._prepopulate // self._instances):
            states, _, _, _ = self._step(env, states, epsilon=1.0)
        env.rmem.flush()

        self._training_loop(duration)

    def _training_loop(self, duration):
        env = VecMonitor(self._vec_env)
        states = env.reset()

        for t in itertools.count(start=1):
            if t > duration:
                return

            if t % self._target_update_freq == 1:
                self._dqn.update_target_net()
                env.rmem.flush()

            if t % self._train_freq == 1:
                minibatch = env.rmem.sample(self._batch_size)
                self._dqn.train(*minibatch)

            epsilon = BaselineDQNAgent.epsilon_schedule(t)
            states, _, _, _ = self._step(env, states, epsilon)

    def _policy(self, states, epsilon):
        assert 0.0 <= epsilon <= 1.0
        # With probability epsilon, take a random action
        if self.action_space.np_random.rand() <= epsilon:
            return [self.action_space.sample() for _ in range(self._instances)]
        # Otherwise, compute the greedy (i.e. best predicted) action
        return self._greedy_actions(states)

    def _greedy_actions(self, states):
        return self._dqn.greedy_actions(states, network='main').numpy()

    def _step(self, vec_env, states, epsilon):
        actions = self._policy(states, epsilon)
        next_states, rewards, dones, infos = vec_env.step(actions)
        return next_states, rewards, dones, infos

    @staticmethod
    def epsilon_schedule(t):
        assert t > 0, "timestep must start at 1"
        epsilon = 1.0 - 0.9 * (t / 1_000_000)
        return max(epsilon, 0.1)
