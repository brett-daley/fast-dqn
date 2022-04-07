from fast_dqn.agents.offpolicy import BaselineOffpolicyAgent, FastOffpolicyAgent


def DQNAgent(make_vec_env_fn, num_envs, **kwargs):
    from fast_dqn.agents.dqn import BaselineDQNAgent, FastDQNAgent
    if num_envs == 1:
        return BaselineDQNAgent(make_vec_env_fn, num_envs, **kwargs)
    return FastDQNAgent(make_vec_env_fn, num_envs, **kwargs)


def DDPGAgent(make_vec_env_fn, num_envs, **kwargs):
    from fast_dqn.agents.ddpg import BaselineDDPGAgent, FastDDPGAgent
    if num_envs == 1:
        return BaselineDDPGAgent(make_vec_env_fn, num_envs, **kwargs)
    return FastDDPGAgent(make_vec_env_fn, num_envs, **kwargs)
