import numpy as np

from fast_dqn.agents.dqn.baseline_dqn_agent import BaselineDQNAgent


class FastDQNAgent(BaselineDQNAgent):
    def __init__(self, make_vec_env_fn, workers, eval_freq, concurrent=True, synchronize=True, **kwargs):
        assert workers >= 1
        if synchronize:
            assert workers != 1
        super().__init__(make_vec_env_fn, workers, eval_freq, **kwargs)

        if synchronize:
            # Target update frequency must be divisible by number of workers to
            # ensure workers use the correct network parameters when synchronized
            assert self._target_update_freq % workers == 0

        assert self._target_update_freq % self._train_freq == 0
        self._minibatches_per_epoch = self._target_update_freq // self._train_freq

        self._concurrent_training = concurrent
        self._synchronize = synchronize

    def _policy(self, states, epsilon):
        assert 0.0 <= epsilon <= 1.0
        # Compute random actions
        N = len(states)
        random_actions = self._random_actions(n=N)
        if epsilon == 1.0:
            return random_actions

        # Compute the greedy (i.e. best predicted) actions
        greedy_actions = np.argmax(self._dqn.predict(states), axis=1)

        # With probability epsilon, take the random action, otherwise greedy
        rng = self.action_space.np_random.rand(N)
        return np.where(rng <= epsilon, random_actions, greedy_actions)
