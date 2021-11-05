from argparse import ArgumentParser
from distutils.util import strtobool
import itertools
import os
os.environ['TF_DETERMINISTIC_OPS'] = '1'
from threading import Thread
from queue import Queue

import numpy as np

from run_dqn import DQNAgent, main, make_parser
from fast_dqn.worker import Worker


class FastDQNAgent(DQNAgent):
    def __init__(self, make_env_fn, workers=8, concurrent=True, synchronize=True, **kwargs):
        assert workers >= 1
        if synchronize:
            assert workers != 1

        envs = tuple(make_env_fn(i) for i in range(workers))

        self.shared_states = np.empty([workers, *envs[0].observation_space.shape], dtype=np.float32)
        self.shared_qvalues = np.empty([workers, envs[0].action_space.n], dtype=np.float32)

        self._workers = tuple(Worker(i, env=envs[i], agent=self) for i in range(workers))

        super().__init__(make_env_fn, **kwargs)
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
        self._prepopulate_replay_memory()
        self._sync_everything()

        for t in itertools.count(start=1):
            if self._evaluate > 0 and t % self._evaluate == 1:
                self._sync_everything()
                mean_perf, std_perf = self.benchmark(epsilon=0.05, episodes=30)
                print("Benchmark (t={}): mean={}, std={}".format(t - 1, mean_perf, std_perf))

            if t > duration:
                self._sync_everything()
                return

            if t % self._target_update_freq == 1:
                self._sync_everything()
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
                self._sync_workers()
                # Compute the Q-values in a single minibatch
                # We use the target network here so we can train the main network in parallel
                self.shared_qvalues[:] = self._dqn.predict_target(self.shared_states).numpy()
            self._workers[i].update(t)

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
        self._env.flush_monitor()
        for w in self._workers:
            for transition in w.flush():
                self._replay_memory.save(*transition)

    def _sync_everything(self):
        self._train_queue.join()
        self._sync_workers()
        self._flush_workers()

    def _step(self, epsilon):
        return self._workers[0]._step(epsilon)


if __name__ == '__main__':
    parser = make_parser()
    parser.add_argument('--concurrent', type=strtobool, default=True)
    parser.add_argument('--workers', type=int, default=8)
    parser.add_argument('--synchronize', type=strtobool, default=True)
    kwargs = vars(parser.parse_args())
    main(FastDQNAgent, kwargs)
