from fast_dqn.agents.dqn.fast_dqn_agent import FastDQNAgent as _FastDQNAgent
from fast_dqn.agents.dqn.baseline_dqn_agent import BaselineDQNAgent as _BaselineDQNAgent


def DQNAgent(make_vec_env_fn, instances, eval_freq, **kwargs):
    if instances == 1:
        return _BaselineDQNAgent(make_vec_env_fn, instances, eval_freq, **kwargs)
    return _FastDQNAgent(make_vec_env_fn, instances, eval_freq, **kwargs)
