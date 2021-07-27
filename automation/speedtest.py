seeds = list(range(5))

yaml = f"""
speedtest:
  prefix: speedtest
  script: experiments/fixed_exploration.py
  kwargs:
    game: pong
    mb-coalescing: [1, 2, 4, 8, 16, 32, 64]
    parallel-training: [False, True]
    timesteps: 1000000
    seed: {seeds}
"""


if __name__ == '__main__':
    import auto_exp
    auto_exp.from_string(yaml)
