from fast_dqn.agents.offpolicy import BaselineOffpolicyAgent
from fast_dqn.agents.dqn.dqn_network import DQNNetwork


class BaselineDQNAgent(BaselineOffpolicyAgent):
    def __init__(self, make_vec_env_fn, num_envs, **kwargs):
        super().__init__(make_vec_env_fn, num_envs, DQNNetwork, **kwargs)

    def policy(self, states, epsilon):
        assert 0.0 <= epsilon <= 1.0
        # With probability epsilon, take a random action
        if self.action_space.np_random.rand() <= epsilon:
            return [self.action_space.sample() for _ in range(self._num_envs)]
        # Otherwise, compute the greedy (i.e. best predicted) action
        return self._greedy_actions(states)

    def epsilon_schedule(self, t):
        assert t > 0, "timestep must start at 1"
        epsilon = 1.0 - 0.9 * (t / 1_000_000)
        return max(epsilon, 0.1)
