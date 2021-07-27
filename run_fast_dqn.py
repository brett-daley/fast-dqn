from argparse import ArgumentParser
from distutils.util import strtobool
import os
from threading import Thread
from queue import Queue

import numpy as np

from run_dqn import DQNAgent, main

os.environ['TF_DETERMINISTIC_OPS'] = '1'


class FastDQNAgent(DQNAgent):
    def __init__(self, env, mb_coalescing=1, **kwargs):
        super().__init__(env, **kwargs)
        assert mb_coalescing >= 1
        self._minibatch_coalescing = mb_coalescing

    def update(self, t, state, action, reward, done):
        assert t > 0, "timestep must start at 1"
        self._replay_memory.save(state, action, reward, done)

        if t % self._target_update_freq == 1:
            self._dqn.update_target_net()

        if t <= self._prepopulate:
            # We're still pre-populating the replay memory
            return

        if t % (self._train_freq * self._minibatch_coalescing) == 1:
            self._train()

    def _train(self):
        batch_size = self._batch_size * self._minibatch_coalescing
        minibatch = self._replay_memory.sample(batch_size)
        self._dqn.train(*minibatch, split=self._minibatch_coalescing)


class ParallelFastDQNAgent(FastDQNAgent):
    def __init__(self, env, **kwargs):
        super().__init__(env, **kwargs)
        self._transition_buffer = []
        self._train_queue = Queue()
        Thread(target=self._train_loop).start()

    def _predict(self, states):
        # We use the target network here so we can train the main network in parallel
        return self._dqn.predict_target(states)

    def update(self, t, state, action, reward, done):
        assert t > 0, "timestep must start at 1"
        self._transition_buffer.append( (state, action, reward, done) )

        if t % self._target_update_freq == 1:
            self._train_queue.join()

            self._dqn.update_target_net()

            for transition in self._transition_buffer:
                self._replay_memory.save(*transition)
                self._transition_buffer.clear()

            if t > self._prepopulate:
                self._train_queue.put_nowait(t)

    def _train_loop(self):
        while True:
            t = self._train_queue.get()
            for i in range(self._target_update_freq):
                if (t + i) % (self._train_freq * self._minibatch_coalescing) == 0:
                    self._train()
            self._train_queue.task_done()


def parse_kwargs():
    parser = ArgumentParser()
    parser.add_argument('--game', type=str, default='pong')
    parser.add_argument('--mb-coalescing', type=int, default=1)
    parser.add_argument('--interp', type=str, default='linear')
    parser.add_argument('--parallel-training', type=strtobool, default=False)
    parser.add_argument('--timesteps', type=int, default=5_000_000)
    parser.add_argument('--seed', type=int, default=0)
    return vars(parser.parse_args())


if __name__ == '__main__':
    kwargs = parse_kwargs()
    parallel_training = kwargs.pop('parallel_training')
    agent_cls = ParallelFastDQNAgent if parallel_training else FastDQNAgent
    main(agent_cls, kwargs)
