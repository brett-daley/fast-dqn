import numpy as np

from fast_dqn.agents.ddpg.ddpg_network import DDPGNetwork
from fast_dqn.agents.offpolicy import BaselineOffpolicyAgent


class BaselineDDPGAgent(BaselineOffpolicyAgent):
    def __init__(self, make_vec_env_fn, num_envs, **kwargs):
        super().__init__(make_vec_env_fn, num_envs, DDPGNetwork, **kwargs)

        self.num_actions = self._vec_env.action_space.shape[-1]
        self.action_limit = self._vec_env.action_space.high.max()

        # DrQv2 prepopulates with 4k steps
        self._prepopulate = 4_000
        self._train_freq = 2
        self._batch_size = 256
        self._target_update_freq = 10_000

    def policy(self, states, stddev):
        # Add gaussian noise to the actions for exploration
        noises = [stddev * np.random.randn(self.num_actions) for _ in range(self._num_envs)]
        actions = np.clip(self._greedy_actions(states) + noises, -self.action_limit, self.action_limit)
        return actions

    # Anneal exploration noise standard deviation
    def epsilon_schedule(self, t):
        assert t > 0, "timestep must start at 1"
        stddev = 1.0 - 0.9 * (t / 100_000)
        return max(stddev, 0.1)
