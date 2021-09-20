seeds = list(range(1))

yaml = f"""
all_games:
  prefix: allgames
  script: run_fast_dqn.py
  kwargs:
    game:
      - alien
      - amidar
      - assault
      - asterix
      - asteroids
      - atlantis
      - bank_heist
      - battle_zone
      - beam_rider
      - bowling
      - boxing
      - breakout
      - centipede
      - chopper_command
      - crazy_climber
      - demon_attack
      - double_dunk
      - enduro
      - fishing_derby
      - freeway
      - frostbite
      - gopher
      - gravitar
      - hero
      - ice_hockey
      - jamesbond
      - kangaroo
      - krull
      - kung_fu_master
      - montezuma_revenge
      - ms_pacman
      - name_this_game
      - pong
      - private_eye
      - qbert
      - riverraid
      - road_runner
      - robotank
      - seaquest
      - space_invaders
      - star_gunner
      - tennis
      - time_pilot
      - tutankham
      - up_n_down
      - venture
      - video_pinball
      - wizard_of_wor
      - zaxxon
    workers: 8
    concurrent: True
    synchronize: True
    timesteps: 50000000
    seed: {seeds}
"""


if __name__ == '__main__':
    import auto_exp
    auto_exp.from_string(yaml)
