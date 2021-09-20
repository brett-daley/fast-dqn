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
        self._make_env_fn = make_env_fn
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
        self._prepopulate_replay_memory()
        assert self._replay_memory._size_now == self._prepopulate
        self._env.enable_monitor(True, auto_flush=True)

        for t in range(1, duration + 1):
            if t % self._target_update_freq == 1:
                self._dqn.update_target_net()

            if t % self._train_freq == 1:
                minibatch = self._replay_memory.sample(self._batch_size)
                self._dqn.train(*minibatch)

            epsilon = DQNAgent.epsilon_schedule(t)
            self._step(epsilon)

        mean_perf, std_perf = self.benchmark(epsilon=0.05, episodes=30)
        print("Agent: mean={}, std={}".format(mean_perf, std_perf))
        mean_perf, std_perf = self.benchmark(epsilon=1.0, episodes=30)
        print("Random: mean={}, std={}".format(mean_perf, std_perf))

        self._shutdown()

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
        next_state, reward, done, _ = self._env.step(action)
        self._replay_memory.save(self._state, action, reward, done)
        self._state = self._env.reset() if done else next_state

    @staticmethod
    def epsilon_schedule(t):
        assert t > 0, "timestep must start at 1"
        epsilon = 1.0 - 0.9 * (t / 1_000_000)
        return max(epsilon, 0.1)

    def _prepopulate_replay_memory(self):
        self._env.enable_monitor(False)
        for _ in range(self._prepopulate):
            self._step(epsilon=1.0)
        self._state = self._env.reset()
        self._env.enable_monitor(True)

    def benchmark(self, epsilon, episodes=30):
        assert episodes > 0
        env = self._make_env_fn()
        env.enable_monitor(False)

        episode_returns = []
        for _ in range(episodes):
            state = env.reset()
            done = False
            total_reward = 0.0

            while not done:
                action = self._policy(state, epsilon)
                state, reward, _, info = env.step(action)
                done = info['real_done']
                total_reward += reward

            episode_returns.append(total_reward)
            total_reward = 0.0
        env.close()

        return np.mean(episode_returns), np.std(episode_returns, ddof=1)

    def _shutdown(self):
        self._env.close()


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
    parser.add_argument('--seed', type=int, default=0)
    return vars(parser.parse_args())


def main(agent_cls, kwargs):
    allow_gpu_memory_growth()

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
