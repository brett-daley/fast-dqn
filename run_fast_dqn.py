from argparse import ArgumentParser
from distutils.util import strtobool
import os
from threading import Thread
from queue import Queue

import numpy as np

from run_dqn import DQNAgent, main

os.environ['TF_DETERMINISTIC_OPS'] = '1'


class FastDQNAgent(DQNAgent):
    def __init__(self, make_env_fn, mb_coalescing=1, concurrent_training=False, workers=1, **kwargs):
        assert workers > 0
        self._workers = tuple(Worker(i, env=make_env_fn(), agent=self) for i in range(workers))

        super().__init__(make_env_fn)
        self._env = self._workers[0]._env
        self._state = self._env.reset()

        assert mb_coalescing >= 1
        self._minibatch_coalescing = mb_coalescing
        assert self._target_update_freq % (self._train_freq * self._minibatch_coalescing) == 0
        self._minibatches_per_epoch = self._target_update_freq // (self._train_freq * self._minibatch_coalescing)

        self._concurrent_training = concurrent_training
        self._train_queue = Queue()
        Thread(target=self._train_loop, daemon=True).start()

    def _predict(self, states):
        # We use the target network here so we can train the main network in parallel
        return self._dqn.predict_target(states)

    def update(self, t):
        assert t > 0, "timestep must start at 1"

        i = (t - 1) % len(self._workers)
        self._workers[i].sample(t)

        if t % self._target_update_freq == 1:
            self._train_queue.join()

            for w in self._workers:
                w.join()
                for transition in w.flush():
                    self._replay_memory.save(*transition)

            self._dqn.update_target_net()

            if t >= self._training_start:
                for _ in range(self._minibatches_per_epoch):
                    self._train_queue.put_nowait(None)
                    if not self._concurrent_training:
                        self._train_queue.join()

    def _train_loop(self):
        while True:
            self._train_queue.get()

            batch_size = self._batch_size * self._minibatch_coalescing
            minibatch = self._replay_memory.sample(batch_size)
            self._dqn.train(*minibatch, split=self._minibatch_coalescing)

            self._train_queue.task_done()


class Worker:
    def __init__(self, worker_id, env, agent):
        self._env = env
        self._state = env.reset()
        self._agent = agent

        self._transition_buffer = []
        self._sample_queue = Queue()
        Thread(target=self._sample_loop, daemon=True).start()

    def sample(self, t):
        self._sample_queue.put_nowait(t)

    def _sample_loop(self):
        while True:
            t = self._sample_queue.get()

            action = self._agent.policy(t, self._state)
            next_state, reward, done, _ = self._env.step(action)
            self._transition_buffer.append( (self._state.copy(), action, reward, done) )
            self._state = self._env.reset() if done else next_state

            self._sample_queue.task_done()

    def join(self):
        self._sample_queue.join()

    def flush(self):
        for transition in self._transition_buffer:
            yield transition
        self._transition_buffer.clear()


def parse_kwargs():
    parser = ArgumentParser()
    parser.add_argument('--game', type=str, default='pong')
    parser.add_argument('--mb-coalescing', type=int, default=1)
    parser.add_argument('--interp', type=str, default='linear')
    parser.add_argument('--concurrent-training', type=strtobool, default=False)
    parser.add_argument('--workers', type=int, default=1)
    parser.add_argument('--timesteps', type=int, default=5_000_000)
    parser.add_argument('--seed', type=int, default=0)
    return vars(parser.parse_args())


if __name__ == '__main__':
    kwargs = parse_kwargs()
    main(FastDQNAgent, kwargs)
