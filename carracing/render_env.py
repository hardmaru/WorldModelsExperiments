# for the purpose of creating visualizations

import numpy as np
import gym

from scipy.misc import imresize as resize
from scipy.misc import toimage as toimage
from gym.spaces.box import Box
from gym.envs.box2d.car_racing import CarRacing

SCREEN_X = 64
SCREEN_Y = 64
FACTOR = 6.25

def _process_frame(frame):
  obs = frame[0:84, :, :].astype(np.float)/255.0
  obs = resize(obs, (64, 64))
  obs = ((1.0 - obs) * 255).round().astype(np.uint8)
  return obs

class CarRacingWrapper(CarRacing):
  def __init__(self):
    super(CarRacingWrapper, self).__init__()
    self.observation_space = Box(low=0, high=255, shape=(SCREEN_X, SCREEN_Y, 3))
    self.custom_viewer = None
    self.frame_count = 0
    self.current_frame = None
    self.vae_frame = None

  def _step(self, action):
    obs, reward, done, _ = super(CarRacingWrapper, self)._step(action)
    self.current_frame = _process_frame(obs)
    return self.current_frame, reward, done, {}

  def _render(self, mode='human', close=False):

    if mode == "state_pixels":
      return super(CarRacingWrapper, self)._render("state_pixels")

    img_orig = self.current_frame

    img_vae = self.vae_frame

    img = np.concatenate((img_orig, img_vae), axis=1)

    img = resize(img, (int(np.round(SCREEN_Y*FACTOR)), int(np.round(SCREEN_X*FACTOR))*2))

    #img = img_orig

    if self.frame_count > 0:
      pass
      #toimage(img, cmin=0, cmax=255).save('output/'+str(self.frame_count)+'.png')
    self.frame_count += 1

    return super(CarRacingWrapper, self)._render(mode=mode, close=close)

def make_env(env_name, seed=-1, render_mode=False):
  env = CarRacingWrapper()
  if (seed >= 0):
    env.seed(seed)
  '''
  print("environment details")
  print("env.action_space", env.action_space)
  print("high, low", env.action_space.high, env.action_space.low)
  print("environment details")
  print("env.observation_space", env.observation_space)
  print("high, low", env.observation_space.high, env.observation_space.low)
  assert False
  '''
  return env
