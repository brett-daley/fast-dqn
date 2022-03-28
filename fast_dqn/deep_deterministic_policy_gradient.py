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

        # TODO: per the original paper (https://arxiv.org/pdf/1509.02971.pdf)
        # "When learning from pixels we used 3 convolutional layers (no pooling) with 
        # 32 filters at each layer"
        def make_encoder():
            return Sequential([
                InputLayer(env.observation_space.shape),
                Conv2D(32, kernel_size=8, strides=4, activation='relu'),
                Conv2D(64, kernel_size=4, strides=2, activation='relu'),
                Conv2D(64, kernel_size=3, strides=1, activation='relu'),
                Flatten(),
            ])
        encoder_output_shape = ((((((env.observation_space.shape[0] - 8) // 4 + 1 ) - 4) // 2 + 1) - 3) // 1 + 1) ** 2 * 64
        def make_actor():
            return Sequential([
                InputLayer(encoder_output_shape),
                Dense(512, activation='relu'),
                Dense(np.prod(env.action_space.shape), activation='tanh')
            ])
        def make_critic():
            return Sequential([
                InputLayer(encoder_output_shape + env.action_space.shape[0]),
                Dense(512, activation='relu'),
                Dense(1)
            ])
        # We make one shared encoder for the actor and critic.
        # TODO: Should we make separate main/target/exec encoders?
        self._encoder = make_encoder()
        self._main_actor = make_actor()
        self._main_critic = make_critic()
        self._actor_vars = self._main_actor.trainable_variables
        self._critic_vars = self._main_critic.trainable_variables
        self._target_actor = make_actor()
        self._target_critic = make_critic()
        self._exec_actor = make_actor()
        self._exec_critic = make_critic()

    def _preprocess_states(self, states):
        return tf.cast(states, tf.float32) / 255.0

    # TODO: naming is a little confusing... this function only gets
    # Called to update the critic network
    @tf.function
    def predict_actions_and_values(self, states, network):
        # Critic does not update encoder
        encoding = tf.stop_gradient(self._encoder(states))
        actions = {
            'main': self._main_actor,
            'target': self._target_actor,
            'exec': self._exec_actor,
        }[network](encoding)
        actions = tf.clip_by_value(actions, -self.action_limit, self.action_limit)
        values = {
            'main': self._main_critic,
            'target': self._target_critic,
            'exec': self._exec_actor,
        }[network](tf.concat([encoding, actions], axis=1))
        return actions, values

    @tf.function
    def predict_actions(self, states, network):
        # Critic does not update encoder
        encoding = tf.stop_gradient(self._encoder(states))
        return {
            'main': self._main_actor,
            'target': self._target_actor,
            'exec': self._exec_actor,
        }[network](encoding)

    @tf.function
    def predict_values(self, states, actions, network):
        encoding = self._encoder(states)
        concat = tf.concat([encoding, actions], axis=1)
        return {
            'main': self._main_critic,
            'target': self._target_critic,
            'exec': self._exec_critic,
        }[network](concat)

    @tf.function
    def train(self, states, actions, rewards, next_states, dones):
        states, next_states = self._preprocess_states(states), self._preprocess_states(next_states)
        next_actions = tf.clip_by_value(self.predict_actions(next_states, network='target'), -self.action_limit, self.action_limit)
        next_Q = self.predict_values(next_states, next_actions, network='target')

        done_mask = 1.0 - tf.cast(dones, tf.float32)
        targets = rewards + self.discount * done_mask * next_Q

        # Critic update
        with tf.GradientTape() as tape:
            Q = self.predict_values(states, actions, network='main')
            # TODO: Should we use huber loss here?
            loss = tf.reduce_mean(huber_loss(targets - Q))

        gradients = tape.gradient(loss, self._critic_vars)
        self.critic_optimizer.apply_gradients(zip(gradients, self._critic_vars))

        # Actor update
        with tf.GradientTape() as tape:
            _, Q = self.predict_actions_and_values(states, network='main')
            loss = - tf.reduce_mean(Q) 

        gradients = tape.gradient(loss, self._actor_vars)
        self.actor_optimizer.apply_gradients(zip(gradients, self._actor_vars))

    def update_target_net(self):
        self._copy_network(self._main_actor.trainable_variables, self._target_actor.trainable_variables)
        self._copy_network(self._main_critic.trainable_variables, self._target_critic.trainable_variables)

    def update_exec_net(self):
        self._copy_network(self._main_actor.trainable_variables, self._exec_actor.trainable_variables)
        self._copy_network(self._main_critic.trainable_variables, self._exec_critic.trainable_variables)

    @tf.function
    def _copy_network(self, src_vars, dst_vars):
        for var, target in zip(src_vars, dst_vars):
            target.assign(var)


def huber_loss(x):
    # Huber loss: This is the correct implementation of DQN's partial derivative clipping
    # See https://github.com/devsisters/DQN-tensorflow/issues/16
    return tf.where(
        tf.abs(x) <= 1.0,    # Condition
        0.5 * tf.square(x),  # True
        tf.abs(x) - 0.5      # False
    )
