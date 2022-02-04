from argparse import ArgumentParser
from distutils.util import strtobool
import os
os.environ['TF_DETERMINISTIC_OPS'] = '1'

import numpy as np
import tensorflow as tf

from fast_dqn import environment
from fast_dqn.agents import DQNAgent


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
    parser.add_argument('--evaluate', type=int, default=250_000)
    parser.add_argument('--num_envs', type=int, default=8)
    parser.add_argument('--seed', type=int, default=0)
    return parser


def main(agent_cls, kwargs):
    allow_gpu_memory_growth()

    seed = kwargs['seed']
    np.random.seed(seed)
    tf.random.set_seed(seed)

    rmem_capacity = 1_000_000
    make_vec_env_fn = lambda instances: environment.make(kwargs['game'], instances, rmem_capacity, seed)

    agent = agent_cls(make_vec_env_fn, kwargs['num_envs'], kwargs['evaluate'], **kwargs)
    agent.run(kwargs['timesteps'])


if __name__ == '__main__':
    parser = make_parser()
    kwargs = vars(parser.parse_args())
    main(DQNAgent, kwargs)
