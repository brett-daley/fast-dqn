from argparse import ArgumentParser
import itertools
import os

from gym.spaces import Discrete
import numpy as np
import tensorflow as tf
from tensorflow.keras.optimizers import RMSprop

import atari_env
from deep_q_network import DeepQNetwork
from replay_memory import ReplayMemory

os.environ['TF_DETERMINISTIC_OPS'] = '1'


class DQNAgent:
    def __init__(self, env, minibatch_coalescing, greedy_action_max_repeat):
        assert isinstance(env.action_space, Discrete)
        assert minibatch_coalescing >= 1
        assert greedy_action_max_repeat >= 0
        self._env = env
        self._minibatch_coalescing = minibatch_coalescing
        self._greedy_action_max_repeat = greedy_action_max_repeat

        optimizer = RMSprop(lr=2.5e-4, rho=0.95, momentum=0.95, epsilon=0.01)
        self._dqn = DeepQNetwork(env, optimizer, discount=0.99)
        self._replay_memory = ReplayMemory(env, capacity=1_000_000)

        self._prepopulate = 50_000
        self._train_freq = 4
        self._batch_size = 32
        self._target_update_freq = 10_000

        self._last_greedy_action = None
        self._time_of_last_greedy_action = 0

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
        Q = self._dqn.predict(state[None])[0]
        self._last_greedy_action = np.argmax(Q)
        self._time_of_last_greedy_action = t
        return self._last_greedy_action

    def _epsilon_schedule(self, t):
        if t <= self._prepopulate:
            return 1.0
        t -= self._prepopulate
        epsilon = 1.0 - 0.9 * (t / 1_000_000)
        return max(epsilon, 0.1)

    def update(self, t, state, action, reward, done):
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
            self._dqn.train(*minibatch, split=self._minibatch_coalescing)


def main(env_id, minibatch_coalescing, greedy_repeat_prob, timesteps, seed):
    np.random.seed(seed)
    tf.random.set_seed(seed)

    env = atari_env.make(env_id)
    env.seed(seed)
    env.action_space.seed(seed)
    state = env.reset()

    agent = DQNAgent(env, minibatch_coalescing, greedy_repeat_prob)

    for t in itertools.count(start=1):
        if t >= timesteps and done:
            env.close()
            break

        action = agent.policy(t, state)
        next_state, reward, done, _ = env.step(action)
        agent.update(t, state, action, reward, done)
        state = env.reset() if done else next_state


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--game', type=str, default='pong')
    parser.add_argument('--coalesce', type=int, default=1)
    parser.add_argument('--greedy-repeat', type=int, default=0)
    parser.add_argument('--timesteps', type=int, default=5_000_000)
    parser.add_argument('--seed', type=int, default=0)
    args = parser.parse_args()
    main(args.game, args.coalesce, args.greedy_repeat, args.timesteps, args.seed)
