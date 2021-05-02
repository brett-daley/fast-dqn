import time

import cv2
from gym.envs.atari.atari_env import AtariEnv
import numpy as np


def run_experiment(dataset, preprocess_fn):
    times = []
    for x in dataset:
        start = time.time()
        y = preprocess_fn(x)
        end = time.time()
        times.append(end - start)

    times = 1e6 * np.asarray(times)
    mean = np.mean(times)
    std = np.std(times, ddof=1)
    print('{:.2f}'.format(mean), '+/-', '{:.2f}'.format(std), 'μs')

    cv2.imwrite('before.jpg', x)
    cv2.imwrite('after.jpg', y)


def main():
    env = AtariEnv('space_invaders', frameskip=4, obs_type='image')
    env.seed(0)
    env.action_space.seed(0)
    state = env.reset()

    dataset = []
    for _ in range(1_000):
        image = cv2.cvtColor(state, cv2.COLOR_BGR2GRAY)
        dataset.append(image)
        image, _, done, _ = env.step(env.action_space.sample())
        if done:
            image = env.reset()

    print("identity", end=': ')
    run_experiment(dataset, lambda x: x)

    for dims in [(84, 84), (80, 84), (80, 104), (80, 80)]:
        for interpolation in [cv2.INTER_LINEAR, cv2.INTER_NEAREST]:
            for crop in [False, True]:
                if crop:
                    preprocess_fn = lambda x: cv2.resize(x[1:-1], dims, interpolation=interpolation)
                else:
                    preprocess_fn = lambda x: cv2.resize(x, dims, interpolation=interpolation)
                print(dims, interpolation, crop, end=': ')
                run_experiment(dataset, preprocess_fn)


if __name__ == '__main__':
    main()
