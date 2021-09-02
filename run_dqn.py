from argparse import ArgumentParser
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
    def __init__(self, make_env_fn, **kwargs):
        self._env = env = make_env_fn()
        assert isinstance(env.action_space, Discrete)
        self._state = env.reset()

        optimizer = RMSprop(lr=2.5e-4, rho=0.95, momentum=0.95, epsilon=0.01)
        self._dqn = DeepQNetwork(env, optimizer, discount=0.99)
        self._replay_memory = ReplayMemory(env, capacity=1_000_000)

        self._prepopulate = 50_000
        self._train_freq = 4
        self._batch_size = 32
        self._target_update_freq = 10_000

    def run(self, duration):
        for _ in range(self._prepopulate):
            self._step(action=self._policy(epsilon=1.0))
        assert self._replay_memory._size_now == self._prepopulate

        for t in range(1, duration + 1):
            if t % self._target_update_freq == 1:
                self._dqn.update_target_net()

            if t % self._train_freq == 1:
                minibatch = self._replay_memory.sample(self._batch_size)
                self._dqn.train(*minibatch)

            epsilon = DQNAgent.epsilon_schedule(t)
            self._step(action=self._policy(epsilon))

        self._env.close()

    def _policy(self, epsilon):
        assert 0.0 <= epsilon <= 1.0
        # With probability epsilon, take a random action
        if np.random.rand() < epsilon:
            return self._env.action_space.sample()

        # Otherwise, compute the greedy (i.e. best predicted) action
        Q = self._dqn.predict(self._state[None])[0]
        return np.argmax(Q)

    def _step(self, action):
        next_state, reward, done, _ = self._env.step(action)
        self._replay_memory.save(self._state, action, reward, done)
        self._state = self._env.reset() if done else next_state

    @staticmethod
    def epsilon_schedule(t):
        assert t > 0, "timestep must start at 1"
        epsilon = 1.0 - 0.9 * (t / 1_000_000)
        return max(epsilon, 0.1)


def parse_kwargs():
    parser = ArgumentParser()
    parser.add_argument('--game', type=str, default='pong')
    parser.add_argument('--interp', type=str, default='linear')
    parser.add_argument('--timesteps', type=int, default=5_000_000)
    parser.add_argument('--seed', type=int, default=0)
    return vars(parser.parse_args())


def main(agent_cls, kwargs):
    seed = kwargs['seed']
    np.random.seed(seed)
    tf.random.set_seed(seed)

    # TODO: Is sharing the seed here ok?
    def make_env_fn():
        env = atari_env.make(kwargs['game'], kwargs['interp'])
        env.seed(seed)
        env.action_space.seed(seed)
        return env

    agent = agent_cls(make_env_fn, **kwargs)
    agent.run(kwargs['timesteps'])


if __name__ == '__main__':
    kwargs = parse_kwargs()
    main(DQNAgent, kwargs)
