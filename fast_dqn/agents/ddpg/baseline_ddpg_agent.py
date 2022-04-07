import itertools

import numpy as np
from tensorflow.keras.optimizers import Adam

from fast_dqn.deep_deterministic_policy_gradient import DDPG
from fast_dqn.environment import VecMonitor


class BaselineDDPGAgent:
    def __init__(self, make_vec_env_fn, instances, **kwargs):
        self._make_vec_env_fn = make_vec_env_fn
        self._instances = instances

        self._vec_env = env = make_vec_env_fn(instances)
        self.num_actions = self._vec_env.action_space.shape[-1]
        self.action_limit = self._vec_env.action_space.high.max()

        actor_optimizer = Adam(lr=1e-4)
        critic_optimizer = Adam(lr=1e-4)
        self._ddpg = DDPG(env, actor_optimizer, critic_optimizer, discount=0.99)

        # DrQv2 prepopulates with 4k steps
        self._prepopulate = 4_000
        self._train_freq = 2
        self._batch_size = 256
        self._target_update_freq = 10_000

    def run(self, duration):
        env = self._vec_env
        states = env.reset()
        assert (self._prepopulate % self._instances) == 0
        for _ in range(self._prepopulate // self._instances):
            states, _, _, _ = self._step(env, states, stddev=1.0)
        env.rmem.flush()

        self._training_loop(duration)

    def _training_loop(self, duration):
        env = VecMonitor(self._vec_env)
        states = env.reset()

        for t in itertools.count(start=1):
            if t > duration:
                return

            if t % self._target_update_freq == 1:
                self._ddpg.update_target_net()
                env.rmem.flush()

            if t % self._train_freq == 1:
                minibatch = env.rmem.sample(self._batch_size)
                self._ddpg.train(*minibatch)

            stddev = BaselineDDPGAgent.stddev_schedule(t)
            states, _, _, _ = self._step(env, states, stddev)

    def _policy(self, states, stddev):
        # Add gaussian noise to the actions for exploration
        noises = [stddev * np.random.randn(self.num_actions) for _ in range(self._instances)]
        actions = np.clip(self._greedy_actions(states) + noises, -self.action_limit, self.action_limit)
        return actions

    def _greedy_actions(self, states):
        return self._ddpg.predict_actions(states, network='main').numpy()

    def _step(self, vec_env, states, stddev):
        actions = self._policy(states, stddev)
        next_states, rewards, dones, infos = vec_env.step(actions)
        return next_states, rewards, dones, infos

    # Anneal exploration noise standard deviation
    @staticmethod
    def stddev_schedule(t):
        assert t > 0, "timestep must start at 1"
        stddev = 1.0 - 0.9 * (t / 100_000)
        return max(stddev, 0.1)
