from fast_dqn.environment import atari_env
from fast_dqn.environment.monitor import VecMonitor
from fast_dqn.environment.replay_memory import ReplayMemory
from fast_dqn.environment.thread_vec_env import ThreadVecEnv


def make(game, instances, seed):
    env_fn = lambda: atari_env.make(game)
    env = ThreadVecEnv([env_fn for _ in range(instances)])
    env.seed(seed)
    # TODO: Change capacity here
    env.replay_memory = ReplayMemory(env, capacity=1_000_000, seed=seed)
    return env
