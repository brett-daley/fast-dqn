from argparse import ArgumentParser
from distutils.util import strtobool
import itertools
import os
from threading import Thread
from queue import Queue

import numpy as np

from run_dqn import DQNAgent, main

os.environ['TF_DETERMINISTIC_OPS'] = '1'


class FastDQNAgent(DQNAgent):
    def __init__(self, make_env_fn, workers=8, concurrent=True, synchronize=True, **kwargs):
        assert workers >= 1
        if synchronize:
            assert workers != 1
        self._workers = tuple(Worker(env=make_env_fn(), agent=self) for _ in range(workers))

        super().__init__(make_env_fn)
        self._env = env = self._workers[0]._env

        assert self._target_update_freq % self._train_freq == 0
        self._minibatches_per_epoch = self._target_update_freq // self._train_freq

        self._concurrent_training = concurrent
        self._synchronize = synchronize
        self._train_queue = Queue()
        Thread(target=self._train_loop, daemon=True).start()

        self._shared_states = np.empty([workers, *env.observation_space.shape], dtype=np.float32)
        self._shared_qvalues = np.empty([workers, env.action_space.n], dtype=np.float32)

    def run(self, duration):
        for t in range(self._prepopulate):
            w = self._workers[t % len(self._workers)]
            w._step(action=self._env.action_space.sample())
        self._flush_workers()
        assert self._replay_memory._size_now == self._prepopulate

        for j in itertools.count():
            for k in range(len(self._workers)):
                t = len(self._workers) * j + k + 1

                if t % self._target_update_freq == 1:
                    self._train_queue.join()
                    self._flush_workers()
                    self._dqn.update_target_net()

                if self._concurrent_training:
                    if t % self._target_update_freq == 1:
                        for _ in range(self._minibatches_per_epoch):
                            self._train_queue.put_nowait(None)
                else:
                    if t % self._train_freq == 1:
                        for w in self._workers:
                            w.join()
                        self._train_queue.put_nowait(None)
                        self._train_queue.join()

                if self._synchronize and k == 0:
                    self._update_worker_q_values()

                self._workers[k].update(t)

                if t >= duration:
                    for w in self._workers:
                        w._env.close()
                    return

    def _train_loop(self):
        while True:
            self._train_queue.get()
            minibatch = self._replay_memory.sample(self._batch_size)
            self._dqn.train(*minibatch)
            self._train_queue.task_done()

    def _flush_workers(self):
        for w in self._workers:
            w.join()
            for transition in w.flush():
                self._replay_memory.save(*transition)

    def _update_worker_q_values(self):
        # TODO: A clearer way to do this would be passing array references to the worker constructor
        # Collect states from the workers
        for i, w in enumerate(self._workers):
            w.join()
            self._shared_states[i] = w.state

        # Compute the Q-values in a single minibatch
        # We use the target network here so we can train the main network in parallel
        self._shared_qvalues = self._dqn.predict_target(self._shared_states)

        # Distribute the Q-values to the workers
        for i, w in enumerate(self._workers):
            w.q_values = self._shared_qvalues[i]

    # These functionalities are deferred to the individual workers
    def _policy(self, t):
        raise NotImplementedError
    def _step(self, action):
        raise NotImplementedError


class Worker:
    def __init__(self, env, agent):
        self._env = env
        self._agent = agent

        self.state = env.reset()
        self.q_values = None

        self._transition_buffer = []
        self._sample_queue = Queue()
        Thread(target=self._sample_loop, daemon=True).start()

    def update(self, t):
        self._sample_queue.put_nowait(t)

    def _sample_loop(self):
        while True:
            t = self._sample_queue.get()
            self._step(action=self._policy(t))
            self._sample_queue.task_done()

    def _policy(self, t):
        assert t > 0, "timestep must start at 1"

        # With probability epsilon, take a random action
        epsilon = self._agent._epsilon_schedule(t)
        if np.random.rand() < epsilon:
            return self._env.action_space.sample()

        # Otherwise, compute the greedy (i.e. best predicted) action
        return np.argmax(self._get_q_values())

    def _get_q_values(self):
        if self.q_values is not None:
            # The agent has pre-computed Q-values for us
            return self.q_values

        # We use the target network here so we can train the main network in parallel
        return self._agent._dqn.predict_target(self.state[None])[0]

    def _step(self, action):
        next_state, reward, done, _ = self._env.step(action)
        self._transition_buffer.append( (self.state.copy(), action, reward, done) )
        self.state = self._env.reset() if done else next_state

    def join(self):
        self._sample_queue.join()

    def flush(self):
        for transition in self._transition_buffer:
            yield transition
        self._transition_buffer.clear()


def parse_kwargs():
    parser = ArgumentParser()
    parser.add_argument('--game', type=str, default='pong')
    parser.add_argument('--interp', type=str, default='linear')
    parser.add_argument('--concurrent', type=strtobool, default=True)
    parser.add_argument('--workers', type=int, default=8)
    parser.add_argument('--synchronize', type=strtobool, default=True)
    parser.add_argument('--timesteps', type=int, default=5_000_000)
    parser.add_argument('--seed', type=int, default=0)
    return vars(parser.parse_args())


if __name__ == '__main__':
    kwargs = parse_kwargs()
    main(FastDQNAgent, kwargs)
