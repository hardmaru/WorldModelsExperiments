import numpy as np
import gym

def make_env(env_name, seed=-1, render_mode=False, load_model=True):
  if env_name == 'doomrnn':
    print('making rnn doom environment')
    from doomrnn import DoomCoverRNNEnv
    env = DoomCoverRNNEnv(render_mode=render_mode, load_model=load_model)
  else:
    print('making real doom environment')
    from doomreal import DoomTakeCoverWrapper
    env = DoomTakeCoverWrapper(render_mode=render_mode, load_model=load_model)
  if (seed >= 0):
    env.seed(seed)
  return env
