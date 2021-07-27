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
        if states.dtype == tf.uint8:
            states = tf.cast(states, tf.float32) / 255.0
        return states

    @tf.function
    def predict(self, states):
        return self._main_net(self._preprocess_states(states))

    @tf.function
    def predict_target(self, states):
        return self._target_net(self._preprocess_states(states))

    @tf.function
    def train(self, states, actions, rewards, next_states, dones, split=1):
        assert states.shape[0] % split == 0
        batch_size = states.shape[0] // split

        next_Q = self.predict_target(next_states)
        targets = rewards + self.discount * (1.0 - dones) * tf.reduce_max(next_Q, axis=1)
        mask = tf.one_hot(actions, depth=next_Q.shape[1])

        for i in range(split):
            sl = slice(i * batch_size, (i + 1) * batch_size)

            with tf.GradientTape() as tape:
                Q = self.predict(states[sl])
                Q = tf.reduce_sum(mask[sl] * Q, axis=1)
                loss = tf.keras.losses.MSE(targets[sl], Q)

            gradients = tape.gradient(loss, self._main_vars)
            self.optimizer.apply_gradients(zip(gradients, self._main_vars))

    @tf.function
    def update_target_net(self):
        for var, target in zip(self._main_vars, self._target_vars):
            target.assign(var)
