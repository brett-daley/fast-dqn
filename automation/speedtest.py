seeds = list(range(5))

yaml = f"""
speedtest:
  prefix: speedtest
  script: experiments/fixed_exploration.py
  kwargs:
    game: pong
    workers: [1, 2, 4, 8, 16, 32]
    concurrent: [False, True]
    synchronized: [False, True]
    timesteps: 500000
    seed: {seeds}
"""


if __name__ == '__main__':
    import auto_exp
    auto_exp.from_string(yaml)
