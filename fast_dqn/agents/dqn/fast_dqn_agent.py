import itertools

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

    def _training_loop(self, duration):
        env = self._vec_env
        states = env.reset()

        for i in itertools.count(start=0):
            start = self._instances * i + 1
            end = start + self._instances

            for t in range(start, end):
                if self._eval_freq > 0 and (t % self._eval_freq) == 1:
                    mean_perf, std_perf = self.evaluate(epsilon=0.05, episodes=30)
                    print("Evaluation (t={}): mean={}, std={}".format(t - 1, mean_perf, std_perf))

                if t > duration:
                    return

                if t % self._target_update_freq == 1:
                    self._dqn.update_target_net()

                if t % self._train_freq == 1:
                    minibatch = env.replay_memory.sample(self._batch_size)
                    self._dqn.train(*minibatch)

            epsilon = BaselineDQNAgent.epsilon_schedule(end)
            states, _, _, _ = self._step(env, states, epsilon)

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
