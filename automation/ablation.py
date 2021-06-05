seeds = list(range(5))

yaml = f"""
baseline:
  prefix: ablation
  script: main.py
  kwargs:
    game: pong
    seed: {seeds}

cache:
  prefix: ablation
  script: main.py
  kwargs:
    game: pong
    cache-size: [10, 100, 1000, 10000, 100000]
    seed: {seeds}

minibatch-coalescing:
  prefix: ablation
  script: main.py
  kwargs:
    game: pong
    mb-coalescing: [2, 4, 8, 16, 32, 64]
    seed: {seeds}

forward-backward-sharing:
  prefix: ablation
  script: main.py
  kwargs:
    game: pong
    fb-sharing: True
    seed: {seeds}

interpolation:
  prefix: ablation
  script: main.py
  kwargs:
    game: pong
    interp: nearest
    seed: {seeds}
"""


if __name__ == '__main__':
    import auto_exp
    auto_exp.from_string(yaml)
