import sys
sys.path.append('.')

import numpy as np
import tensorflow as tf

from fast_dqn import atari_env
from run_dqn import train
from run_fast_dqn import FastDQNAgent, ParallelFastDQNAgent, parse_kwargs


def main(agent_cls, kwargs):
    # All of this setup is the same as run_dqn.py
    game = kwargs['game']
    seed = kwargs['seed']
    interpolation = kwargs['interp']
    timesteps = kwargs['timesteps']

    np.random.seed(seed)
    tf.random.set_seed(seed)

    env = atari_env.make(game, interpolation)
    env.seed(seed)
    env.action_space.seed(seed)

    agent = agent_cls(env, **kwargs)

    # Here we intercept the agent's exploration parameters
    agent._training_start = 10_000
    agent._epsilon_schedule = lambda t: 0.1

    train(env, agent, timesteps)


if __name__ == '__main__':
    kwargs = parse_kwargs()
    parallel_training = kwargs.pop('parallel_training')
    agent_cls = ParallelFastDQNAgent if parallel_training else FastDQNAgent
    main(agent_cls, kwargs)
