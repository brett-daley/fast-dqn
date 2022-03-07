from argparse import ArgumentParser
from distutils.util import strtobool
import os
os.environ['TF_DETERMINISTIC_OPS'] = '1'

import numpy as np
import tensorflow as tf

from fast_dqn import environment
from fast_dqn.agents import DQNAgent
from fast_dqn.environment.replay_memory import ReplayMemory


def allow_gpu_memory_growth():
    try:
        gpu_list = tf.config.list_physical_devices('GPU')
    except AttributeError:
        gpu_list = tf.config.experimental.list_physical_devices('GPU')

    for gpu in gpu_list:
        tf.config.experimental.set_memory_growth(gpu, True)


def make_parser():
    parser = ArgumentParser()
    parser.add_argument('--game', type=str, default='pong')
    parser.add_argument('--timesteps', type=int, default=5_000_000)
    # Evaluation disabled by default. Use 250k to match DeepMind Nature paper.
    parser.add_argument('--evaluate', type=int, default=0)
    parser.add_argument('--num_envs', type=int, default=8)
    parser.add_argument('--seed', type=int, default=0)
    return parser


def main(agent_cls, kwargs):
    allow_gpu_memory_growth()

    seed = kwargs['seed']
    np.random.seed(seed)
    tf.random.set_seed(seed)

    num_envs = kwargs['num_envs']
    assert (1_000_000 % num_envs) == 0
    rmem_capacity = 1_000_000 // num_envs
    def rmem_fn():
        rmem = ReplayMemory(rmem_capacity)
        rmem.seed(seed)
        return rmem

    make_vec_env_fn = lambda instances: environment.make(kwargs['game'], instances, rmem_fn, seed)

    agent = agent_cls(make_vec_env_fn, num_envs, kwargs['evaluate'], **kwargs)
    agent.run(kwargs['timesteps'])


if __name__ == '__main__':
    parser = make_parser()
    kwargs = vars(parser.parse_args())
    main(DQNAgent, kwargs)
