import itertools
from threading import Thread
from queue import Queue

import numpy as np

from fast_dqn.agents.dqn.baseline_dqn_agent import BaselineDQNAgent
from fast_dqn.environment import VecMonitor


# Put tuples of the form (function, args) in this queue to call them asynchronously in a background thread.
# Warning: Any values returned by the functions cannot be retrieved.
async_queue = Queue()

def _queue_runner():
    while True:
        function, args = async_queue.get()
        function(*args)
        async_queue.task_done()
Thread(target=_queue_runner, daemon=True).start()


class FastDQNAgent(BaselineDQNAgent):
    def __init__(self, make_vec_env_fn, instances, **kwargs):
        super().__init__(make_vec_env_fn, instances, **kwargs)
        self._exec_update_freq = 10_000

    def _training_loop(self, duration):
        env = VecMonitor(self._vec_env)
        states = env.reset()

        for i in itertools.count(start=0):
            start = self._instances * i + 1
            end = start + self._instances

            for t in range(start, end):
                if t > duration:
                    async_queue.join()
                    return

                # Periodically update the target/exec networks
                update_target_net = (t % self._target_update_freq == 1)
                update_exec_net = (t % self._exec_update_freq == 1)
                if update_target_net or update_exec_net:
                    if update_exec_net:
                        self._dqn.update_exec_net()
                    async_queue.join()  # Wait for all training operations to finish
                    if update_target_net:
                        self._dqn.update_target_net()
                        env.rmem.flush()

                if t % self._train_freq == 1:
                    minibatch = env.rmem.sample(self._batch_size)
                    async_queue.put_nowait((self._dqn.train, minibatch))

            epsilon = BaselineDQNAgent.epsilon_schedule(end)
            states, _, _, _ = self._step(env, states, epsilon)

    def _greedy_actions(self, states):
        return self._dqn.greedy_actions(states, network='exec').numpy()
