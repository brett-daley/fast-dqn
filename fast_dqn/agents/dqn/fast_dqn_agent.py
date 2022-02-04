import itertools

import numpy as np

from fast_dqn.agents.dqn.baseline_dqn_agent import BaselineDQNAgent
from fast_dqn.environment import VecMonitor


class FastDQNAgent(BaselineDQNAgent):
    def _training_loop(self, duration):
        env = VecMonitor(self._vec_env)
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
                    minibatch = env.sample_replay_memory(self._batch_size)
                    self._dqn.train(*minibatch)

            epsilon = BaselineDQNAgent.epsilon_schedule(end)
            states, _, _, _ = self._step(env, states, epsilon)
