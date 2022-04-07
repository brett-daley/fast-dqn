from fast_dqn.agents.dqn.baseline_dqn_agent import BaselineDQNAgent
from fast_dqn.agents.offpolicy import FastOffpolicyAgent


# Python's method resolution order first inherits from the abstract, fast off-policy agent.
# Then, it inherits the policy and exploration schedule from the baseline DQN agent.
# This is all we need to define the fast DQN agent!
class FastDQNAgent(FastOffpolicyAgent, BaselineDQNAgent):
    pass
