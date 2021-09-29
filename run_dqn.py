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
    def __init__(self, make_env_fn, evaluate, **kwargs):
        self._make_env_fn = make_env_fn
        self._env = env = make_env_fn(0)
        assert isinstance(env.action_space, Discrete)
        self._state = env.reset()

        self._evaluate = evaluate
        if evaluate > 0:
            self._benchmark_env = make_env_fn(0)
            self._benchmark_env.enable_monitor(False)

        optimizer = RMSprop(lr=2.5e-4, rho=0.95, epsilon=0.01, centered=True)
        self._dqn = DeepQNetwork(env, optimizer, discount=0.99)
        # TODO: We shouldn't hardcode history_len
        self._replay_memory = ReplayMemory(env, capacity=1_000_000, history_len=4, seed=kwargs['seed'])

        self._prepopulate = 50_000
        self._train_freq = 4
        self._batch_size = 32
        self._target_update_freq = 10_000

    def run(self, duration):
        self._prepopulate_replay_memory()
        self._env.enable_monitor(True, auto_flush=True)

        for t in itertools.count(start=1):
            if self._evaluate > 0 and t % self._evaluate == 1:
                    mean_perf, std_perf = self.benchmark(epsilon=0.05, episodes=30)
                    print("Benchmark (t={}): mean={}, std={}".format(t - 1, mean_perf, std_perf))

            if t > duration:
                return

            if t % self._target_update_freq == 1:
                self._dqn.update_target_net()

            if t % self._train_freq == 1:
                minibatch = self._replay_memory.sample(self._batch_size)
                self._dqn.train(*minibatch)

            epsilon = DQNAgent.epsilon_schedule(t)
            self._step(epsilon)

    def _policy(self, state, epsilon):
        assert 0.0 <= epsilon <= 1.0
        # With probability epsilon, take a random action
        if self._env.action_space.np_random.rand() <= epsilon:
            return self._env.action_space.sample()

        # Otherwise, compute the greedy (i.e. best predicted) action
        Q = self._dqn.predict(state[None])[0]
        return np.argmax(Q)

    def _step(self, epsilon):
        action = self._policy(self._state, epsilon)
        next_state, reward, done, info = self._env.step(action)
        self._replay_memory.save(self._state, action, reward, done)
        self._state = self._env.reset() if done else next_state
        return next_state, reward, done, info

    @staticmethod
    def epsilon_schedule(t):
        assert t > 0, "timestep must start at 1"
        epsilon = 1.0 - 0.9 * (t / 1_000_000)
        return max(epsilon, 0.1)

    def _prepopulate_replay_memory(self):
        self._env.enable_monitor(False)
        for _ in range(self._prepopulate):
            _, _, _, info = self._step(epsilon=1.0)

        # Finish the current episode so we can start training with a new one.
        # (We do this to avoid accidentally bootstrapping from the next episode.)
        # Warning: This will get stuck in a loop if the episode never terminates!
        # Our time-limit wrapper ensures that this doesn't happen.
        while not info['real_done']:
            _, _, _, info = self._step(epsilon=1.0)

        self._env.enable_monitor(True)

    def benchmark(self, epsilon, episodes=30):
        assert episodes > 0
        env = self._benchmark_env

        for _ in range(episodes):
            state = env.reset()
            done = False

            while not done:
                action = self._policy(state, epsilon)
                state, _, _, info = env.step(action)
                done = info['real_done']

        returns = env.get_episode_returns()[-episodes:]
        return np.mean(returns), np.std(returns, ddof=1)


def allow_gpu_memory_growth():
    try:
        gpu_list = tf.config.list_physical_devices('GPU')
    except AttributeError:
        gpu_list = tf.config.experimental.list_physical_devices('GPU')

    for gpu in gpu_list:
        tf.config.experimental.set_memory_growth(gpu, True)


def parse_kwargs():
    parser = ArgumentParser()
    parser.add_argument('--game', type=str, default='pong')
    parser.add_argument('--interp', type=str, default='linear')
    parser.add_argument('--timesteps', type=int, default=5_000_000)
    parser.add_argument('--evaluate', type=int, default=250_000)
    parser.add_argument('--seed', type=int, default=0)
    return vars(parser.parse_args())


def main(agent_cls, kwargs):
    allow_gpu_memory_growth()

    seed = kwargs['seed']
    np.random.seed(seed)
    tf.random.set_seed(seed)

    def make_env_fn(value_added_to_seed):
        env = atari_env.make(kwargs['game'], kwargs['interp'])
        env.seed(seed + value_added_to_seed)
        env.action_space.seed(seed + value_added_to_seed)
        return env

    agent = agent_cls(make_env_fn, **kwargs)
    agent.run(kwargs['timesteps'])


if __name__ == '__main__':
    kwargs = parse_kwargs()
    main(DQNAgent, kwargs)
