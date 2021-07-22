from argparse import ArgumentParser
from distutils.util import strtobool
import os

import numpy as np

from run_dqn import DQNAgent, main

os.environ['TF_DETERMINISTIC_OPS'] = '1'


class FastDQNAgent(DQNAgent):
    def __init__(self, env, mb_coalescing=1, fb_sharing=False, **kwargs):
        super().__init__(env, **kwargs)

        assert mb_coalescing >= 1
        self._minibatch_coalescing = mb_coalescing

        self._fb_sharing = fb_sharing
        self._precomputed_action = None

    def policy(self, t, state):
        assert t > 0, "timestep must start at 1"
        epsilon = self._epsilon_schedule(t)

        # With probability epsilon, take a random action
        if np.random.rand() < epsilon:
            return self._env.action_space.sample()
        # If we have a precomputed action ready, take it
        if self._fb_sharing and (self._precomputed_action is not None):
            return self._precomputed_action
        # Otherwise, compute the greedy (i.e. best predicted) action
        return self._greedy_action(state)

    def update(self, t, state, action, reward, done, next_state):
        assert t > 0, "timestep must start at 1"
        self._replay_memory.save(state, action, reward, done)

        if t % self._target_update_freq == 1:
            self._dqn.update_target_net()

        if t <= self._prepopulate:
            # We're still pre-populating the replay memory
            return

        if t % (self._train_freq * self._minibatch_coalescing) == 1:
            batch_size = self._batch_size * self._minibatch_coalescing
            minibatch = self._replay_memory.sample(batch_size)
            minibatch = self._replay_memory.insert_most_recent(minibatch, next_state)
            self._precomputed_action = self._dqn.train(*minibatch, split=self._minibatch_coalescing)
        else:
            self._precomputed_action = None

        if done:
            self._precomputed_action = None


def parse_kwargs():
    parser = ArgumentParser()
    parser.add_argument('--game', type=str, default='pong')
    parser.add_argument('--mb-coalescing', type=int, default=1)
    parser.add_argument('--interp', type=str, default='linear')
    parser.add_argument('--fb-sharing', type=strtobool, default=False)
    parser.add_argument('--timesteps', type=int, default=5_000_000)
    parser.add_argument('--seed', type=int, default=0)
    return vars(parser.parse_args())


if __name__ == '__main__':
    kwargs = parse_kwargs()
    main(FastDQNAgent, kwargs)
