from fast_dqn.agents.ddpg.baseline_ddpg_agent import BaselineDDPGAgent
from fast_dqn.agents.offpolicy import FastOffpolicyAgent


# Python's method resolution order first inherits from the abstract, fast off-policy agent.
# Then, it inherits the policy and exploration schedule from the baseline DDPG agent.
# This is all we need to define the fast DDPG agent!
class FastDDPGAgent(FastOffpolicyAgent, BaselineDDPGAgent):
    pass
