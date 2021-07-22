from argparse import ArgumentParser
from distutils.util import strtobool
import itertools
import os

from gym.spaces import Discrete
import numpy as np
import tensorflow as tf
from tensorflow.keras.optimizers import RMSprop

from fast_dqn import atari_env
from fast_dqn.deep_q_network import DeepQNetwork
from fast_dqn.replay_memory import ReplayMemory

os.environ['TF_DETERMINISTIC_OPS'] = '1'


class DQNAgent:
    def __init__(self, env, mb_coalescing=1,
                 fb_sharing=False, **kwargs):
        assert isinstance(env.action_space, Discrete)
        assert mb_coalescing >= 1
        self._env = env
        self._minibatch_coalescing = mb_coalescing

        optimizer = RMSprop(lr=2.5e-4, rho=0.95, momentum=0.95, epsilon=0.01)
        self._dqn = DeepQNetwork(env, optimizer, discount=0.99)
        self._replay_memory = ReplayMemory(env, capacity=1_000_000)

        self._prepopulate = 50_000
        self._train_freq = 4
        self._batch_size = 32
        self._target_update_freq = 10_000

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

    def _greedy_action(self, state):
        Q = self._dqn.predict(state[None])[0]
        return np.argmax(Q)

    def _epsilon_schedule(self, t):
        if t <= self._prepopulate:
            return 1.0
        t -= self._prepopulate
        epsilon = 1.0 - 0.9 * (t / 1_000_000)
        return max(epsilon, 0.1)

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


def train(env, agent, timesteps):
    state = env.reset()

    for t in itertools.count(start=1):
        if t >= timesteps and done:
            env.close()
            break

        action = agent.policy(t, state)
        next_state, reward, done, _ = env.step(action)
        agent.update(t, state, action, reward, done, next_state)
        state = env.reset() if done else next_state


def main(kwargs):
    game = kwargs['game']
    seed = kwargs['seed']
    interpolation = kwargs['interp']
    timesteps = kwargs['timesteps']

    np.random.seed(seed)
    tf.random.set_seed(seed)

    env = atari_env.make(game, interpolation)
    env.seed(seed)
    env.action_space.seed(seed)

    agent = DQNAgent(env, **kwargs)
    train(env, agent, timesteps)


if __name__ == '__main__':
    kwargs = parse_kwargs()
    main(kwargs)
