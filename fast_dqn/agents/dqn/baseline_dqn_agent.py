import itertools

import numpy as np
from tensorflow.keras.optimizers import RMSprop

from fast_dqn import atari_env
from fast_dqn.deep_q_network import DeepQNetwork
from fast_dqn.replay_memory import ReplayMemory


class BaselineDQNAgent:
    def __init__(self, make_vec_env_fn, instances, eval_freq, **kwargs):
        self._make_vec_env_fn = make_vec_env_fn
        self._instances = instances

        self._vec_env = env = make_vec_env_fn(instances)
        self.action_space = self._vec_env.action_space

        self._eval_freq = eval_freq

        optimizer = RMSprop(lr=2.5e-4, rho=0.95, epsilon=0.01, centered=True)
        self._dqn = DeepQNetwork(env, optimizer, discount=0.99)
        self._replay_memory = ReplayMemory(env, capacity=1_000_000, seed=kwargs['seed'])

        self._prepopulate = 50_000
        self._train_freq = 4
        self._batch_size = 32
        self._target_update_freq = 10_000

    def run(self, duration):
        eval_env = self._eval_vec_env = self._make_vec_env_fn(instances=1)
        eval_env.silence_monitor(True)

        prepop_env = self._make_vec_env_fn(self._instances)
        prepop_env.silence_monitor(True)
        states = prepop_env.reset()
        assert (self._prepopulate % self._instances) == 0
        for _ in range(self._prepopulate // self._instances):
            states, _, _, _ = self._step(prepop_env, states, epsilon=1.0)

        self._training_loop(duration)

    def _training_loop(self, duration):
        env = self._vec_env
        states = env.reset()

        for t in itertools.count(start=1):
            if self._eval_freq > 0 and (t % self._eval_freq) == 1:
                mean_perf, std_perf = self.evaluate(epsilon=0.05, episodes=30)
                print("Evaluation (t={}): mean={}, std={}".format(t - 1, mean_perf, std_perf))

            if t > duration:
                return

            if t % self._target_update_freq == 1:
                self._dqn.update_target_net()

            if t % self._train_freq == 1:
                minibatch = self._replay_memory.sample(self._batch_size)
                self._dqn.train(*minibatch)

            epsilon = BaselineDQNAgent.epsilon_schedule(t)
            states, _, _, _ = self._step(env, states, epsilon)

    def _policy(self, states, epsilon):
        assert 0.0 <= epsilon <= 1.0
        # With probability epsilon, take a random action
        if self.action_space.np_random.rand() <= epsilon:
            return self._random_actions(n=1)
        # Otherwise, compute the greedy (i.e. best predicted) action
        return np.argmax(self._dqn.predict(states), axis=1)

    def _random_actions(self, n):
        return np.stack([self.action_space.sample() for _ in range(n)])

    def _step(self, vec_env, states, epsilon):
        actions = self._policy(states, epsilon)
        next_states, rewards, dones, infos = vec_env.step(actions)
        self._replay_memory.save(states, actions, rewards, dones)
        return next_states, rewards, dones, infos

    @staticmethod
    def epsilon_schedule(t):
        assert t > 0, "timestep must start at 1"
        epsilon = 1.0 - 0.9 * (t / 1_000_000)
        return max(epsilon, 0.1)

    def evaluate(self, epsilon, episodes=30):
        assert episodes > 0
        env = self._eval_vec_env

        for _ in range(episodes):
            state = env.reset()
            done = False

            while not done:
                action = self._policy(state, epsilon)
                state, _, _, info = env.step(action)
                done = info[0]['real_done']

        returns = env.get_episode_returns()[-episodes:]
        return np.mean(returns), np.std(returns, ddof=1)
