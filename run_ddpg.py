from argparse import ArgumentParser
from distutils.util import strtobool
import os
os.environ['TF_DETERMINISTIC_OPS'] = '1'

import numpy as np
import tensorflow as tf

from fast_dqn import environment
from fast_dqn.agents import DDPGAgent
from run_dqn import allow_gpu_memory_growth


def make_parser():
    parser = ArgumentParser()
    # Deepmind Control Suite environments require both a domain and a task
    parser.add_argument('--domain', type=str, default='cheetah')
    parser.add_argument('--task', type=str, default='run')
    parser.add_argument('--timesteps', type=int, default=5_000_000)
    parser.add_argument('--num_envs', type=int, default=8)
    parser.add_argument('--seed', type=int, default=0)
    return parser


def main(agent_cls, kwargs):
    allow_gpu_memory_growth()

    seed = kwargs['seed']
    np.random.seed(seed)
    tf.random.set_seed(seed)

    rmem_capacity = 1_000_000

    make_env_fn = lambda instances: environment.make_dmc_env(kwargs['domain'], kwargs['task'], instances, rmem_capacity, seed)

    agent = agent_cls(make_env_fn, **kwargs)
    agent.run(kwargs['timesteps'])


if __name__ == '__main__':
    parser = make_parser()
    kwargs = vars(parser.parse_args())
    main(DDPGAgent, kwargs)
