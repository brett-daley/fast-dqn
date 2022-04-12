from abc import ABC, abstractmethod

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, Dense, Flatten, InputLayer


class OffpolicyNetwork(ABC):
    def __init__(self, env, optimizer_fn, discount):
        # Networks must be defined in subclass:
        self._main_net = None
        self._target_net = None
        self._exec_net = None  # Used by Fast DRL for concurrent training/execution

    def get_model(self, network):
        return {
            'main': self._main_net,
            'target': self._target_net,
            'exec': self._exec_net,
        }[network]

    def _preprocess_states(self, states):
        if states.dtype == tf.uint8:
            return tf.cast(states, tf.float32) / 255.0
        return tf.cast(states, tf.float32)

    @tf.function
    def predict(self, states, network):
        model = self.get_model(network)
        return model(self._preprocess_states(states))

    @abstractmethod
    def greedy_actions(self, states, network):
        raise NotImplementedError

    @abstractmethod
    def train(self, states, actions, rewards, next_states, dones):
        raise NotImplementedError

    @tf.function
    def update_target_net(self):
        copy_network(from_network=self._main_net, to_network=self._target_net)

    @tf.function
    def update_exec_net(self):
        copy_network(from_network=self._main_net, to_network=self._exec_net)


def copy_network(from_network, to_network):
    from_vars = from_network.trainable_variables
    to_vars = to_network.trainable_variables
    assert len(from_vars) == len(to_vars)
    for var, target in zip(from_vars, to_vars):
        target.assign(var)
