from fast_dqn.environment import atari_env
from fast_dqn.environment.monitor import VecMonitor, Monitor
from fast_dqn.environment.replay_memory import ReplayMemory
from fast_dqn.environment.thread_vec_env import ThreadVecEnv

import dmc2gym

# TODO: This make is for atari
def make(game, instances, rmem_capacity, seed):
    replay_memory = ReplayMemory(rmem_capacity, instances, seed)
    env_fn = lambda: atari_env.make(game)
    env = ThreadVecEnv([env_fn for _ in range(instances)], replay_memory)
    env.seed(seed)
    return env

# TODO: and this make is for deepmind control
def make_dmc_env(domain, task, instances, rmem_capacity, seed):
    replay_memory = ReplayMemory(rmem_capacity, instances, seed)
    env_fn = lambda: atari_env.HistoryWrapper(Monitor(dmc2gym.make(domain_name=domain,
                                                            task_name=task,
                                                            from_pixels=True,
                                                            visualize_reward=False,
                                                            channels_first=False,
                                                            time_limit=1000,
                                                            )
                                            ), 
                                    history_len=3)
    env = ThreadVecEnv([env_fn for _ in range(instances)], replay_memory)
    return env
