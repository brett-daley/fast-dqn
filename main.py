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
from fast_dqn.fifo_cache import FIFOCache
from fast_dqn.replay_memory import ReplayMemory

os.environ['TF_DETERMINISTIC_OPS'] = '1'


class DQNAgent:
    def __init__(self, env, mb_coalescing=1, greedy_repeat=0,
                 cache_size=0, fb_sharing=False, **kwargs):
        assert isinstance(env.action_space, Discrete)
        assert mb_coalescing >= 1
        assert greedy_repeat >= 0
        self._env = env
        self._minibatch_coalescing = mb_coalescing
        self._greedy_action_max_repeat = greedy_repeat

        optimizer = RMSprop(lr=2.5e-4, rho=0.95, momentum=0.95, epsilon=0.01)
        self._dqn = DeepQNetwork(env, optimizer, discount=0.99)
        self._replay_memory = ReplayMemory(env, capacity=1_000_000)

        self._prepopulate = 50_000
        self._train_freq = 4
        self._batch_size = 32
        self._target_update_freq = 10_000

        self._last_greedy_action = None
        self._time_of_last_greedy_action = 0

        self._fb_sharing = fb_sharing
        self._next_action = None

        self._cache = FIFOCache(cache_size) if (cache_size > 0) else None

    def policy(self, t, state):
        assert t > 0, "timestep must start at 1"
        epsilon = self._epsilon_schedule(t)

        # With probability epsilon, take a random action
        if np.random.rand() < epsilon:
            return self._env.action_space.sample()
        # Else, if the timer hasn't expired, take the previous greedy action
        elif (t - self._time_of_last_greedy_action) <= self._greedy_action_max_repeat:
            if self._last_greedy_action is not None:
                return self._last_greedy_action

        # Otherwise, take the predicted best action (greedy)
        if self._fb_sharing and (self._next_action is not None):
            self._last_greedy_action = self._next_action
        else:
            self._last_greedy_action = self._greedy_action(state)

        self._time_of_last_greedy_action = t
        return self._last_greedy_action

    def _greedy_action(self, state):
        if self._cache is not None:
            try:
                return self._cache[state]
            except KeyError:
                pass

        Q = self._dqn.predict(state[None])[0]
        action = np.argmax(Q)

        if self._cache is not None:
            self._cache.push(state, action)
        return action

    def _epsilon_schedule(self, t):
        if t <= self._prepopulate:
            return 1.0
        t -= self._prepopulate
        epsilon = 1.0 - 0.9 * (t / 1_000_000)
        return max(epsilon, 0.1)

    def update(self, t, state, action, reward, done, next_state):
        assert t > 0, "timestep must start at 1"
        self._replay_memory.save(state, action, reward, done)

        if done:
            self._last_greedy_action = None

        if t % self._target_update_freq == 1:
            self._dqn.update_target_net()

        if t <= self._prepopulate:
            # We're still pre-populating the replay memory
            return

        if t % (self._train_freq * self._minibatch_coalescing) == 1:
            batch_size = self._batch_size * self._minibatch_coalescing
            minibatch = self._replay_memory.sample(batch_size)

            minibatch = self._replay_memory.insert_most_recent(minibatch, next_state)

            self._next_action = self._dqn.train(*minibatch, split=self._minibatch_coalescing)

        else:
            self._next_action = None

        if done:
            self._next_action = None


def parse_kwargs():
    parser = ArgumentParser()
    parser.add_argument('--game', type=str, default='pong')
    parser.add_argument('--mb-coalescing', type=int, default=1)
    parser.add_argument('--cache-size', type=int, default=0)
    parser.add_argument('--greedy-repeat', type=int, default=0)
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
