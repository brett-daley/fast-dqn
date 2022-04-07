import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, Dense, Flatten, InputLayer

from fast_dqn.agents.offpolicy.offpolicy_network import OffpolicyNetwork


class DQNNetwork(OffpolicyNetwork):
    def __init__(self, env, optimizer_fn, discount):
        self.optimizer = optimizer_fn()
        self.discount = discount

        def model():
            return Sequential([
                InputLayer(env.observation_space.shape),
                Conv2D(32, kernel_size=8, strides=4, activation='relu'),
                Conv2D(64, kernel_size=4, strides=2, activation='relu'),
                Conv2D(64, kernel_size=3, strides=1, activation='relu'),
                Flatten(),
                Dense(512, activation='relu'),
                Dense(env.action_space.n),
            ])
        self._main_net = model()
        self._target_net = model()
        self._exec_net = model()

    @tf.function
    def greedy_actions(self, states, network):
        Q = self.predict(states, network)
        return tf.argmax(Q, axis=1)

    @tf.function
    def train(self, states, actions, rewards, next_states, dones):
        next_Q = self.predict(next_states, network='target')
        done_mask = 1.0 - tf.cast(dones, tf.float32)
        targets = rewards + self.discount * done_mask * tf.reduce_max(next_Q, axis=1)

        action_mask = tf.one_hot(actions, depth=next_Q.shape[1])

        with tf.GradientTape() as tape:
            Q = self.predict(states, network='main')
            Q = tf.reduce_sum(action_mask * Q, axis=1)
            loss = tf.reduce_mean(tf.square(targets - Q))

        main_vars = self._main_net.trainable_variables
        gradients = tape.gradient(loss, main_vars)
        self.optimizer.apply_gradients(zip(gradients, main_vars))
