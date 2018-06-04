'''
saves ~ 200 episodes generated from a random policy
'''


import numpy as np
import random
import os
import config
import gym
from doomreal import _process_frame
from env import make_env
from model import make_model

MAX_FRAMES = 2100 # from doomtakecover
MAX_TRIALS = 200 # just use this to extract one trial. 
MIN_LENGTH = 100

render_mode = False # for debugging.

DIR_NAME = 'record'
if not os.path.exists(DIR_NAME):
    os.makedirs(DIR_NAME)

model = make_model(config.games['doomreal'])

total_frames = 0
model.make_env(render_mode=render_mode, load_model=False) # random weights
for trial in range(MAX_TRIALS): # 200 trials per worker
  try:
    random_generated_int = random.randint(0, 2**31-1)
    filename = DIR_NAME+"/"+str(random_generated_int)+".npz"
    recording_obs = []
    recording_action = []

    np.random.seed(random_generated_int)
    model.env.seed(random_generated_int)

    # random policy
    # model.init_random_model_params(stdev=0.2)
    # more diverse random policy, works slightly better:
    repeat = np.random.randint(1, 11)

    obs = model.env.reset() # the latent code
    pixel_obs = model.env.current_obs # secret 64x64 obs frame

    if obs is None:
      obs = np.zeros(model.input_size)

    for frame in range(MAX_FRAMES):
      if render_mode:
        model.env.render("human")
      #action = model.get_action(obs) # use more diverse random policy:
      if frame % repeat == 0:
        action = np.random.rand() * 2.0 - 1.0
        repeat = np.random.randint(1, 11)
      recording_obs.append(pixel_obs)
      recording_action.append(action)
      obs, reward, done, info = model.env.step(action)
      pixel_obs = model.env.current_obs # secret 64x64 obs frame

      if done:
        break

    total_frames += frame
    print("dead at", frame, "total recorded frames for this worker", total_frames)
    recording_obs = np.array(recording_obs, dtype=np.uint8)
    recording_action = np.array(recording_action, dtype=np.float16)
    if (len(recording_obs) > MIN_LENGTH):
      np.savez_compressed(filename, obs=recording_obs, action=recording_action)
  except gym.error.Error:
    print("stupid doom error, life goes on")
    model.env.close()
    model.make_env(render_mode=render_mode)
    continue
model.env.close()
