from fast_dqn.agents.dqn.fast_dqn_agent import FastDQNAgent as _FastDQNAgent
from fast_dqn.agents.dqn.baseline_dqn_agent import BaselineDQNAgent as _BaselineDQNAgent


def DQNAgent(make_vec_env_fn, num_envs, **kwargs):
    if num_envs == 1:
        return _BaselineDQNAgent(make_vec_env_fn, num_envs, **kwargs)
    return _FastDQNAgent(make_vec_env_fn, num_envs, **kwargs)
