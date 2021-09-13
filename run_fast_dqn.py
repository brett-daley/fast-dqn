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

        envs = tuple(make_env_fn() for _ in range(workers))

        self.shared_states = np.empty([workers, *envs[0].observation_space.shape], dtype=np.float32)
        self.shared_qvalues = np.empty([workers, envs[0].action_space.n], dtype=np.float32)
        # Set permissions to avoid accidental writes
        self.shared_states.flags.writeable = True
        self.shared_qvalues.flags.writeable = False

        self._workers = tuple(Worker(i, env=envs[i], agent=self) for i in range(workers))

        super().__init__(make_env_fn)
        self._env = env = self._workers[0]._env

        if synchronize:
            # Target update frequency must be divisible by number of workers to
            # ensure workers use the correct network parameters when synchronized
            assert self._target_update_freq % workers == 0

        assert self._target_update_freq % self._train_freq == 0
        self._minibatches_per_epoch = self._target_update_freq // self._train_freq

        self._concurrent_training = concurrent
        self._synchronize = synchronize
        self._train_queue = Queue()
        Thread(target=self._train_loop, daemon=True).start()

    def run(self, duration):
        for t in range(self._prepopulate):
            w = self._workers[t % len(self._workers)]
            w._step(epsilon=1.0)
        self._sync_workers()
        self._flush_workers()
        assert self._replay_memory._size_now == self._prepopulate

        for t in itertools.count(start=1):
            if t % self._target_update_freq == 1:
                self._train_queue.join()
                self._sync_workers()
                self._flush_workers()
                self._dqn.update_target_net()

                if self._concurrent_training:
                    for _ in range(self._minibatches_per_epoch):
                        self._train_queue.put_nowait(None)

            if not self._concurrent_training:
                if t % self._train_freq == 1:
                    self._sync_workers()
                    self._train_queue.put_nowait(None)
                    self._train_queue.join()

            i = t % len(self._workers)
            if i == 1 and self._synchronize:
                self._update_worker_qvalues()
            self._workers[i].update(t)

            if t >= duration:
                mean_perf, std_perf = self.benchmark(epsilon=0.05, episodes=30)
                print("Agent: mean={}, std={}".format(mean_perf, std_perf))
                mean_perf, std_perf = self.benchmark(epsilon=1.0, episodes=30)
                print("Random: mean={}, std={}".format(mean_perf, std_perf))

                self._shutdown()
                return

    def _train_loop(self):
        while True:
            self._train_queue.get()
            minibatch = self._replay_memory.sample(self._batch_size)
            self._dqn.train(*minibatch)
            self._train_queue.task_done()

    def _sync_workers(self):
        for w in self._workers:
            w.join()

    def _flush_workers(self):
        for w in self._workers:
            for transition in w.flush():
                self._replay_memory.save(*transition)

    def _update_worker_qvalues(self):
        self._sync_workers()

        # Toggle read-only
        for a in [self.shared_states, self.shared_qvalues]:
            a.flags.writeable = not a.flags.writeable

        # Compute the Q-values in a single minibatch
        # We use the target network here so we can train the main network in parallel
        self.shared_qvalues = self._dqn.predict_target(self.shared_states).numpy()

        # Toggle read-only
        for a in [self.shared_states, self.shared_qvalues]:
            a.flags.writeable = not a.flags.writeable

    def _shutdown(self):
        self._train_queue.join()
        self._sync_workers()
        for w in self._workers:
            w.close()

    def _policy(self, state, epsilon):
        return self._workers[0].policy(state, epsilon)

    def _step(self, epsilon):
        # This functionality is deferred to the individual workers
        raise NotImplementedError


class Worker:
    def __init__(self, worker_id, env, agent):
        self._id = worker_id
        self._env = env
        self._agent = agent
        self._np_random = np.random.RandomState(seed=0)

        self._state = env.reset()

        self._transition_buffer = []
        self._sample_queue = Queue()
        Thread(target=self._sample_loop, daemon=True).start()

    def update(self, t):
        self._sample_queue.put_nowait(t)

    def _sample_loop(self):
        while True:
            t = self._sample_queue.get()
            epsilon = DQNAgent.epsilon_schedule(t)
            self._step(epsilon)
            self._sample_queue.task_done()

    def policy(self, state, epsilon):
        assert 0.0 <= epsilon <= 1.0
        # With probability epsilon, take a random action
        if self._np_random.rand() < epsilon:
            return self._env.action_space.sample()

        # Otherwise, compute the greedy (i.e. best predicted) action
        return np.argmax(self._get_qvalues(state))

    @property
    def _state(self):
        return self._agent.shared_states[self._id]

    @_state.setter
    def _state(self, state):
        self._agent.shared_states[self._id] = state

    def _get_qvalues(self, state):
        if self._agent._synchronize:
            # The agent has pre-computed Q-values for us
            return self._agent.shared_qvalues[self._id]
        else:
            # We use the target network here so we can train the main network in parallel
            return self._agent._dqn.predict_target(state[None])[0]

    def _step(self, epsilon):
        action = self.policy(self._state, epsilon)
        next_state, reward, done, _ = self._env.step(action)
        self._transition_buffer.append( (self._state.copy(), action, reward, done) )
        self._state = self._env.reset() if done else next_state

    def join(self):
        self._sample_queue.join()

    def flush(self):
        for transition in self._transition_buffer:
            yield transition
        self._transition_buffer.clear()

    def close(self):
        self._env.close()


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
