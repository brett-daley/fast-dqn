from argparse import ArgumentParser
from distutils.util import strtobool
import os
os.environ['TF_DETERMINISTIC_OPS'] = '1'

import numpy as np

from fast_dqn import environment
from fast_dqn.agents import DQNAgent
from fast_dqn.environment.replay_memory import ReplayMemory


def allow_gpu_memory_growth():
    import tensorflow as tf
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
    parser.add_argument('--num_envs', type=int, default=8)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--pytorch', action='store_true')
    return parser


def main(agent_cls, kwargs):
    seed = kwargs['seed']
    np.random.seed(seed)

    if kwargs['pytorch']:
        import torch
        torch.manual_seed(seed)
    else:
        import tensorflow as tf
        tf.random.set_seed(seed)
        allow_gpu_memory_growth()

    rmem_capacity = 1_000_000
    make_vec_env_fn = lambda instances: environment.make(kwargs['game'], instances, rmem_capacity, seed)

    agent = agent_cls(make_vec_env_fn, **kwargs)
    agent.run(kwargs['timesteps'])


if __name__ == '__main__':
    parser = make_parser()
    kwargs = vars(parser.parse_args())
    main(DQNAgent, kwargs)
