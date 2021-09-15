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
    def __init__(self, make_env_fn, workers=8, concurrent=True, synchronized=True, **kwargs):
        assert workers >= 1
        if synchronized:
            assert workers != 1

        envs = tuple(make_env_fn() for _ in range(workers))

        self.shared_states = np.empty([workers, *envs[0].observation_space.shape], dtype=np.float32)
        self.shared_qvalues = np.empty([workers, envs[0].action_space.n], dtype=np.float32)
        # Set permissions to avoid accidental writes
        self.shared_states.flags.writeable = True
        self.shared_qvalues.flags.writeable = False

        super().__init__(make_env_fn)

        self._workers = tuple(
            Worker(i, envs[i], self._dqn, self.shared_states, self.shared_qvalues, seed=kwargs['seed'], synchronized=synchronized)
            for i in range(workers)
        )

        self._env = env = self._workers[0]._env

        if synchronized:
            # Target update frequency must be divisible by number of workers to
            # ensure workers use the correct network parameters when synchronized
            assert self._target_update_freq % workers == 0

        assert self._target_update_freq % self._train_freq == 0
        self._minibatches_per_epoch = self._target_update_freq // self._train_freq

        self._concurrent_training = concurrent
        self._synchronized_sampling = synchronized
        self._train_queue = Queue()
        Thread(target=self._train_loop, daemon=True).start()

    def run(self, duration):
        for t in range(self._prepopulate):
            w = self._workers[t % len(self._workers)]
            w.step(epsilon=1.0)
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
            if i == 1 and self._synchronized_sampling:
                self._update_worker_qvalues()
            epsilon = DQNAgent.epsilon_schedule(t)
            self._workers[i].step(epsilon)

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

    def _step(self, epsilon):
        # This functionality is deferred to the individual workers
        raise NotImplementedError


class Worker:
    def __init__(self, worker_id, env, dqn, shared_states, shared_qvalues, seed, synchronized):
        self._env = env
        self._queue = Queue()
        self._transition_buffer = []

        # TODO: Use unique seeds here
        np_random = np.random.RandomState(seed=0)

        Thread(
            target=worker_task, daemon=True,
            args=(worker_id, env, dqn, self._queue, self._transition_buffer, shared_states, shared_qvalues, np_random, synchronized)
        ).start()

    def step(self, epsilon):
        self._queue.put_nowait(epsilon)

    def join(self):
        self._queue.join()

    def flush(self):
        for transition in self._transition_buffer:
            yield transition
        self._transition_buffer.clear()


def worker_task(worker_id, env, dqn, task_queue, transition_buffer, shared_states, shared_qvalues, np_random, synchronized):
    def policy(state, epsilon):
        assert 0.0 <= epsilon <= 1.0
        # With probability epsilon, take a random action
        if np_random.rand() < epsilon:
            return env.action_space.sample()

        # Otherwise, compute the greedy (i.e. best predicted) action
        if synchronized:
            # The main process has pre-computed Q-values for us
            qvalues = shared_qvalues[worker_id]
        else:
            # We must compute our own Q-values
            # Use the target network here so we can train the main network concurrently
            qvalues = dqn.predict_target(state[None])[0]
        return np.argmax(qvalues)

    state = env.reset()
    while True:
        epsilon = task_queue.get()

        action = policy(state, epsilon)
        next_state, reward, done, _ = env.step(action)
        transition_buffer.append( (state.copy(), action, reward, done) )
        state = env.reset() if done else next_state

        task_queue.task_done()


def parse_kwargs():
    parser = ArgumentParser()
    parser.add_argument('--game', type=str, default='pong')
    parser.add_argument('--interp', type=str, default='linear')
    parser.add_argument('--concurrent', type=strtobool, default=True)
    parser.add_argument('--workers', type=int, default=8)
    parser.add_argument('--synchronized', type=strtobool, default=True)
    parser.add_argument('--timesteps', type=int, default=5_000_000)
    parser.add_argument('--seed', type=int, default=0)
    return vars(parser.parse_args())


if __name__ == '__main__':
    kwargs = parse_kwargs()
    main(FastDQNAgent, kwargs)
