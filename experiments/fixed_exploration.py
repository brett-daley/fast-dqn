import sys
sys.path.append('.')

from run_fast_dqn import FastDQNAgent as _FastDQNAgent
from run_fast_dqn import main, parse_kwargs


# Override the agent so we can inject new exploration parameters
class FastDQNAgent(_FastDQNAgent):
    def run(self, duration):
        self._evaluate = False
        # Here we intercept the agent's exploration parameters
        self._prepopulate = 2
        self._epsilon_schedule = lambda t: 0.1
        super().run(duration)


if __name__ == '__main__':
    kwargs = parse_kwargs()
    main(FastDQNAgent, kwargs)
