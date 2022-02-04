from fast_dqn.environment import atari_env
from fast_dqn.environment.monitor import VecMonitor
from fast_dqn.environment.thread_vec_env import ThreadVecEnv


def make(game, instances, rmem_capacity, seed):
    env_fn = lambda: atari_env.make(game)
    env = ThreadVecEnv([env_fn for _ in range(instances)], rmem_capacity)
    env.seed(seed)
    return env
