import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, Dense, Flatten, InputLayer


class DeepQNetwork:
    def __init__(self, env, optimizer, discount):
        self.optimizer = optimizer
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

        self._main_vars = self._main_net.trainable_variables
        self._target_vars = self._target_net.trainable_variables

    def _preprocess_states(self, states):
        return tf.cast(states, tf.float32) / 255.0

    @tf.function
    def predict(self, states):
        return self._main_net(self._preprocess_states(states))

    @tf.function
    def predict_target(self, states):
        return self._target_net(self._preprocess_states(states))

    @tf.function
    def train(self, states, actions, rewards, next_states, dones):
        next_Q = self.predict_target(next_states)
        done_mask = 1.0 - tf.cast(dones, tf.float32)
        targets = rewards + self.discount * done_mask * tf.reduce_max(next_Q, axis=1)

        action_mask = tf.one_hot(actions, depth=next_Q.shape[1])

        with tf.GradientTape() as tape:
            Q = self.predict(states)
            Q = tf.reduce_sum(action_mask * Q, axis=1)
            loss = tf.reduce_mean(huber_loss(targets - Q))

        gradients = tape.gradient(loss, self._main_vars)
        self.optimizer.apply_gradients(zip(gradients, self._main_vars))

    @tf.function
    def update_target_net(self):
        for var, target in zip(self._main_vars, self._target_vars):
            target.assign(var)


def huber_loss(x):
    # Huber loss: This is the correct implementation of DQN's partial derivative clipping
    # See https://github.com/devsisters/DQN-tensorflow/issues/16
    return tf.where(
        tf.abs(x) <= 1.0,    # Condition
        0.5 * tf.square(x),  # True
        tf.abs(x) - 0.5      # False
    )
