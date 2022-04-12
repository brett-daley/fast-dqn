import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, Dense, Flatten, InputLayer

from fast_dqn.agents.offpolicy.offpolicy_network import OffpolicyNetwork


class DDPGNetwork(OffpolicyNetwork):
    def __init__(self, env, optimizer_fn, discount):
        self.actor_optimizer = optimizer_fn()
        self.critic_optimizer = optimizer_fn()
        self.discount = discount
        self.action_limit = env.action_space.high.max()

        self._main_net = _DDPGSharedFeaturesModel(env)
        self._target_net = _DDPGSharedFeaturesModel(env)
        self._exec_net = _DDPGSharedFeaturesModel(env)

    # TODO: naming is a little confusing... this function only gets
    # Called to update the critic network
    @tf.function
    def predict_actions_and_values(self, states, network):
        model = self.get_model(network)
        encoding = model.encode(self._preprocess_states(states))
        # Actor does not update encoder
        encoding = tf.stop_gradient(encoding)
        actions = model.actions(encoding)
        actions = tf.clip_by_value(actions, -self.action_limit, self.action_limit)
        values = model.values(encoding, actions)
        return actions, values

    @tf.function
    def greedy_actions(self, states, network):
        model = self.get_model(network)
        encoding = model.encode(self._preprocess_states(states))
        # Actor does not update encoder
        encoding = tf.stop_gradient(encoding)
        return model.actions(encoding)

    @tf.function
    def predict_values(self, states, actions, network):
        model = self.get_model(network)
        encoding = model.encode(self._preprocess_states(states))
        actions = tf.cast(actions, encoding.dtype)
        return model.values(encoding, actions)

    @tf.function
    def train(self, states, actions, rewards, next_states, dones):
        _, next_Q = self.predict_actions_and_values(next_states, network='target')

        done_mask = 1.0 - tf.cast(dones, tf.float32)
        targets = rewards + self.discount * done_mask * tf.squeeze(next_Q)

        # Critic Update
        with tf.GradientTape() as tape:
            Q = self.predict_values(states, actions, network='main')
            Q = tf.squeeze(Q)
            loss = tf.reduce_mean(tf.square(targets - Q))

        # Critic vars implicitly update encoder too
        gradient = tape.gradient(loss, self._main_net.critic_vars)
        self.critic_optimizer.apply_gradients(zip(gradient, self._main_net.critic_vars))

        # Actor Update
        with tf.GradientTape() as tape:
            _, Q = self.predict_actions_and_values(states, network='main')
            loss = - tf.reduce_mean(Q) 

        gradient = tape.gradient(loss, self._main_net.actor_vars)
        self.actor_optimizer.apply_gradients(zip(gradient, self._main_net.actor_vars))


class _DDPGSharedFeaturesModel(tf.keras.Model):
    def __init__(self, env):
        super().__init__(self)
        kernel_size = 3
        num_filters = 32

        self._encoder = Sequential([
            InputLayer(env.observation_space.shape),
            Conv2D(num_filters, kernel_size=kernel_size, strides=2, activation='relu'),
            Conv2D(num_filters, kernel_size=kernel_size, strides=1, activation='relu'),
            Conv2D(num_filters, kernel_size=kernel_size, strides=1, activation='relu'),
            Conv2D(num_filters, kernel_size=kernel_size, strides=1, activation='relu'),
            Flatten(),
        ])
        encoder_output_shape = ((((((((env.observation_space.shape[0] - kernel_size) // 2 + 1 ) - kernel_size) + 1) - kernel_size) + 1) - kernel_size) + 1) ** 2 * num_filters

        self._actor = Sequential([
            InputLayer(encoder_output_shape),
            Dense(256, activation='relu'),
            Dense(256, activation='relu'),
            Dense(np.prod(env.action_space.shape), activation='tanh')
        ])

        self._critic = Sequential([
            InputLayer(encoder_output_shape + env.action_space.shape[0]),
            Dense(256, activation='relu'),
            Dense(256, activation='relu'),
            Dense(1)
        ])

        self.critic_vars = self._encoder.trainable_variables + self._critic.trainable_variables
        # Actor does not update encoder
        self.actor_vars = self._actor.trainable_variables

    def call(self, inputs):
        raise NotImplementedError

    def encode(self, states):
        return self._encoder(states)

    def values(self, encoding, actions):
        concat = tf.concat([encoding, actions], axis=1)
        return self._critic(concat)

    def actions(self, encoding):
        return self._actor(encoding)
