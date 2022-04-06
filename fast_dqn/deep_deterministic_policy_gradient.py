import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, Dense, Flatten, InputLayer
import numpy as np


class DDPG:
    def __init__(self, env, actor_optimizer, critic_optimizer, discount):
        self.actor_optimizer = actor_optimizer
        self.critic_optimizer = critic_optimizer
        self.discount = discount
        self.action_limit = env.action_space.high.max()

        kernel_size = 3
        num_filters = 32
        def make_encoder():
            return Sequential([
                InputLayer(env.observation_space.shape),
                Conv2D(num_filters, kernel_size=kernel_size, strides=2, activation='relu'),
                Conv2D(num_filters, kernel_size=kernel_size, strides=1, activation='relu'),
                Conv2D(num_filters, kernel_size=kernel_size, strides=1, activation='relu'),
                Conv2D(num_filters, kernel_size=kernel_size, strides=1, activation='relu'),
                Flatten(),
            ])
        encoder_output_shape = ((((((((env.observation_space.shape[0] - kernel_size) // 2 + 1 ) - kernel_size) + 1) - kernel_size) + 1) - kernel_size) + 1) ** 2 * num_filters

        def make_actor():
            return Sequential([
                InputLayer(encoder_output_shape),
                Dense(256, activation='relu'),
                Dense(256, activation='relu'),
                Dense(np.prod(env.action_space.shape), activation='tanh')
            ])
        def make_critic():
            return Sequential([
                InputLayer(encoder_output_shape + env.action_space.shape[0]),
                Dense(256, activation='relu'),
                Dense(256, activation='relu'),
                Dense(1)
            ])

        self._main_encoder = make_encoder()
        self._main_actor = make_actor()
        self._main_critic = make_critic()
        self._encoder_vars = self._main_encoder.trainable_variables
        self._actor_vars = self._main_actor.trainable_variables
        self._critic_vars = self._main_critic.trainable_variables
        self._target_encoder = make_encoder()
        self._target_actor = make_actor()
        self._target_critic = make_critic()
        self._exec_encoder = make_encoder()
        self._exec_actor = make_actor()
        self._exec_critic = make_critic()

    def _preprocess_states(self, states):
        return tf.cast(states, tf.float32) / 255.0

    # TODO: naming is a little confusing... this function only gets
    # Called to update the critic network
    @tf.function
    def predict_actions_and_values(self, states, network):
        states = self._preprocess_states(states)
        encoding = {
            'main': self._main_encoder,
            'target': self._target_encoder,
            'exec': self._exec_encoder,
        }[network](states)
        # Actor does not update encoder
        encoding = tf.stop_gradient(encoding)
        actions = {
            'main': self._main_actor,
            'target': self._target_actor,
            'exec': self._exec_actor,
        }[network](encoding)
        actions = tf.clip_by_value(actions, -self.action_limit, self.action_limit)
        values = {
            'main': self._main_critic,
            'target': self._target_critic,
            'exec': self._exec_critic,
        }[network](tf.concat([encoding, actions], axis=1))
        return actions, values

    @tf.function
    def predict_actions(self, states, network):
        states = self._preprocess_states(states)
        encoding = {
            'main': self._main_encoder,
            'target': self._target_encoder,
            'exec': self._exec_encoder,
        }[network](states)
        # Actor does not update encoder
        encoding = tf.stop_gradient(encoding)
        return {
            'main': self._main_actor,
            'target': self._target_actor,
            'exec': self._exec_actor,
        }[network](encoding)

    @tf.function
    def predict_values(self, states, actions, network):
        states = self._preprocess_states(states)
        encoding = {
            'main': self._main_encoder,
            'target': self._target_encoder,
            'exec': self._exec_encoder,
        }[network](states)
        concat = tf.concat([encoding, actions], axis=1)
        return {
            'main': self._main_critic,
            'target': self._target_critic,
            'exec': self._exec_critic,
        }[network](concat)

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

        gradient = tape.gradient(loss, self._encoder_vars + self._critic_vars)
        self.critic_optimizer.apply_gradients(zip(gradient, self._encoder_vars + self._critic_vars))

        # Actor Update
        with tf.GradientTape() as tape:
            _, Q = self.predict_actions_and_values(states, network='main')
            loss = - tf.reduce_mean(Q) 

        gradient = tape.gradient(loss, self._actor_vars)
        self.actor_optimizer.apply_gradients(zip(gradient, self._actor_vars))

    def update_target_net(self):
        self._copy_network(self._main_encoder.trainable_variables, self._target_encoder.trainable_variables)
        self._copy_network(self._main_actor.trainable_variables, self._target_actor.trainable_variables)
        self._copy_network(self._main_critic.trainable_variables, self._target_critic.trainable_variables)

    def update_exec_net(self):
        self._copy_network(self._main_encoder.trainable_variables, self._exec_encoder.trainable_variables)
        self._copy_network(self._main_actor.trainable_variables, self._exec_actor.trainable_variables)
        self._copy_network(self._main_critic.trainable_variables, self._exec_critic.trainable_variables)

    @tf.function
    def _copy_network(self, src_vars, dst_vars):
        for var, target in zip(src_vars, dst_vars):
            target.assign(var)
