from abc import ABC, abstractmethod
import itertools

from tensorflow.keras.optimizers import Adam

from fast_dqn.environment import VecMonitor


class BaselineOffpolicyAgent(ABC):
    def __init__(self, make_vec_env_fn, num_envs, network_cls, **kwargs):
        self._make_vec_env_fn = make_vec_env_fn
        self._num_envs = num_envs

        self._vec_env = env = make_vec_env_fn(num_envs)
        self.action_space = self._vec_env.action_space

        optimizer_fn = lambda: Adam(lr=1e-4)
        self.net = network_cls(env, optimizer_fn, discount=0.99)

        self._prepopulate = 50_000
        self._train_freq = 4
        self._batch_size = 32
        self._target_update_freq = 10_000

    def run(self, duration):
        env = self._vec_env
        states = env.reset()
        assert (self._prepopulate % self._num_envs) == 0
        for _ in range(self._prepopulate // self._num_envs):
            states, _, _, _ = self._step(env, states, epsilon=1.0)
        env.rmem.flush()

        self._training_loop(duration)

    def _training_loop(self, duration):
        env = VecMonitor(self._vec_env)
        states = env.reset()

        for t in itertools.count(start=1):
            if t > duration:
                return

            if t % self._target_update_freq == 1:
                self.net.update_target_net()
                env.rmem.flush()

            if t % self._train_freq == 1:
                minibatch = env.rmem.sample(self._batch_size)
                self.net.train(*minibatch)

            epsilon = self.epsilon_schedule(t)
            states, _, _, _ = self._step(env, states, epsilon)

    @abstractmethod
    def policy(self, states, epsilon):
        raise NotImplementedError

    def _greedy_actions(self, states):
        return self.net.greedy_actions(states, network='main').numpy()

    def _step(self, vec_env, states, epsilon):
        actions = self.policy(states, epsilon)
        next_states, rewards, dones, infos = vec_env.step(actions)
        return next_states, rewards, dones, infos

    @abstractmethod
    def epsilon_schedule(self, t):
        raise NotImplementedError
