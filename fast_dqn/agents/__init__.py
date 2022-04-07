from fast_dqn.agents.dqn.fast_dqn_agent import FastDQNAgent as _FastDQNAgent
from fast_dqn.agents.dqn.baseline_dqn_agent import BaselineDQNAgent as _BaselineDQNAgent


def DQNAgent(make_vec_env_fn, num_envs, **kwargs):
    if num_envs == 1:
        return _BaselineDQNAgent(make_vec_env_fn, num_envs, **kwargs)
    return _FastDQNAgent(make_vec_env_fn, num_envs, **kwargs)


from fast_dqn.agents.ddpg.fast_ddpg_agent import FastDDPGAgent as _FastDDPGAgent
from fast_dqn.agents.ddpg.baseline_ddpg_agent import BaselineDDPGAgent as _BaselineDDPGAgent

def DDPGAgent(make_vec_env_fn, num_envs, **kwargs):
    if num_envs == 1:
        return _BaselineDDPGAgent(make_vec_env_fn, num_envs, **kwargs)
    return _FastDDPGAgent(make_vec_env_fn, num_envs, **kwargs)
