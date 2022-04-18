from argparse import ArgumentParser
import dmc2gym
import numpy as np
import time
import os
import torch
import torch.nn as nn

from fast_dqn.environment import atari_env

from stable_baselines3.dqn import DQN
from stable_baselines3.ddpg import DDPG
from stable_baselines3.common.noise import NormalActionNoise
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from stable_baselines3.common.vec_env.vec_transpose import VecTransposeImage
from stable_baselines3.common.vec_env.subproc_vec_env import SubprocVecEnv
from stable_baselines3.common.vec_env.vec_frame_stack import VecFrameStack
from stable_baselines3.common.vec_env.vec_monitor import VecMonitor
from stable_baselines3.common.monitor import ResultsWriter


class CustomCNN(BaseFeaturesExtractor):
    """
    Create a Feature Extractor to match the one we use for fast-ddpg
    :param observation_space: (gym.Space)
    :param features_dim: (int) Number of features extracted.
        This corresponds to the number of units for the last layer.
    """

    def __init__(self, observation_space, features_dim = 1024):
        super().__init__(observation_space, features_dim)
        kernel_size = 3
        num_filters = 32
        self.cnn = nn.Sequential(
            nn.Conv2d(observation_space.shape[0], num_filters, kernel_size=kernel_size, stride=2),
            nn.ReLU(),
            nn.Conv2d(num_filters, num_filters, kernel_size=kernel_size, stride=1),
            nn.ReLU(),
            nn.Conv2d(num_filters, num_filters, kernel_size=kernel_size, stride=1),
            nn.ReLU(),
            nn.Conv2d(num_filters, num_filters, kernel_size=kernel_size, stride=1),
            nn.ReLU(),
            nn.Flatten()
        )
        self._features_dim = ((((((((observation_space.shape[1] - kernel_size) // 2 + 1) - kernel_size) + 1) - kernel_size) + 1) - kernel_size) + 1) ** 2 * num_filters

    def forward(self, observations):
        return self.cnn(observations)


def make_parser():
    parser = ArgumentParser()
    parser.add_argument('model', choices=['DQN', 'DDPG'])
    parser.add_argument('--game', type=str, default='pong')
    # Deepmind Control Suite environments require both a domain and a task
    parser.add_argument('--domain', type=str, default='cartpole')
    parser.add_argument('--task', type=str, default='balance')
    parser.add_argument('--timesteps', type=int, default=5_000_000)
    parser.add_argument('--num_envs', type=int, default=8)
    parser.add_argument('--seed', type=int, default=0)
    return parser


def make_env(kwargs):
    env = {
        "DQN": VecTransposeImage(VecFrameStack(
                            SubprocVecEnv(
                                [lambda: atari_env.make(kwargs['game'])
                                for _ in range(kwargs['num_envs'])]), 
                        n_stack=4)),
        "DDPG": VecFrameStack(
                    SubprocVecEnv(
                        [lambda: dmc2gym.make(domain_name=kwargs['domain'], 
                                        task_name=kwargs['task'], 
                                        from_pixels=True, 
                                        visualize_reward=False, 
                                        time_limit=1000, 
                                        channels_first=False)
                        for _ in range(kwargs['num_envs'])]), 
                        n_stack=3)
    }[kwargs['model']]
    return VecMonitor(env, filename=None)


def make_net(model_name, env, rmem_capacity, seed):
    if model_name == 'DQN':
        model = DQN("CnnPolicy",
                env,
                buffer_size=rmem_capacity,
                learning_starts=0,
                seed=seed,
        )
        prepopulate(model, 50_000)
    elif model_name == 'DDPG':
        # The noise objects for DDPG
        n_actions = env.action_space.shape[-1]
        action_noise = NormalActionNoise(mean=np.zeros(n_actions), sigma=0.1 * np.ones(n_actions))

        policy_kwargs = dict(
            net_arch=[256, 256],
            features_extractor_class=CustomCNN
        )
        model = DDPG("CnnPolicy",
                env,
                buffer_size=rmem_capacity,
                learning_starts=0,
                action_noise=action_noise,
                train_freq=(2, "step"),
                gradient_steps=1,
                learning_rate=1e-4,
                policy_kwargs=policy_kwargs,
                seed=seed,
            )
        prepopulate(model, 4_000)
    return model


def prepopulate(model, prepopulate_timesteps):
    """Prepopulate the replay buffer"""
    prepop_timesteps, prepop_callback = model._setup_learn(
        total_timesteps=prepopulate_timesteps,
        eval_env=None,
        eval_freq=-1,
    )
    while model.num_timesteps < prepop_timesteps:
        model.collect_rollouts(
            model.env,
            callback=prepop_callback,
            train_freq=model.train_freq,
            replay_buffer=model.replay_buffer,
            learning_starts=prepop_timesteps,
            log_interval=None
        )


def main(kwargs):
    env_name = kwargs['game'] if kwargs['model'] == "DQN" else f"{kwargs['domain']}_{kwargs['task']}"
    print(f"Running {kwargs['model']} on {env_name}")

    seed = kwargs['seed']
    np.random.seed(seed)
    torch.manual_seed(seed)

    rmem_capacity = 1_000_000

    env = make_env(kwargs)
    # make_net also takes care of prepopulating the replay buffer
    model = make_net(kwargs['model'], env, rmem_capacity, seed)    

    resultsdir = f"results/{env_name}"
    if not os.path.exists(resultsdir):
        os.makedirs(resultsdir)

    env.results_writer = ResultsWriter(os.path.join(resultsdir, f"sb3_num_envs={kwargs['num_envs']}_seed={seed}"), header={"t_start": env.t_start, "env_id": f"{env_name}"})
    # Reset the monitor's start time to after we prepopulate the buffer
    env.t_start = time.time()
    
    model.learn(total_timesteps=kwargs['timesteps'], log_interval=None, reset_num_timesteps=True)


if __name__ == '__main__':
    parser = make_parser()
    kwargs = vars(parser.parse_args())
    main(kwargs)
