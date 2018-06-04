# for the purpose of creating visualizations

import numpy as np
import gym
import os
import json

from scipy.misc import imresize as resize
from scipy.misc import toimage as toimage
from gym.spaces.box import Box
from gym.utils import seeding

SCREEN_X = 64
SCREEN_Y = 64
FACTOR = 8

with open(os.path.join('initial_z', 'initial_z.json'), 'r') as f:
  [initial_mu, initial_logvar] = json.load(f)

initial_mu_logvar = [list(elem) for elem in zip(initial_mu, initial_logvar)]

def get_pi_idx(x, pdf):
  # samples from a categorial distribution
  N = pdf.size
  accumulate = 0
  for i in range(0, N):
    accumulate += pdf[i]
    if (accumulate >= x):
      return i
  print('error with sampling ensemble')
  return -1

class CarRacingDream(gym.Env):
  metadata = {
      'render.modes': ['human', 'rgb_array'],
      'video.frames_per_second' : 60
  }

  def __init__(self, agent):
    self.observation_space = Box(low=-50., high=50., shape=(32)) # , dtype=np.float32
    self._seed()
    self.agent = agent
    self.vae = agent.vae
    self.rnn = agent.rnn
    self.z_size = self.rnn.hps.output_seq_width
    self.viewer = None
    self.frame_count = None
    self.z = None
    self.temperature = 0.7
    self.vae_frame = None
    self._reset()

  def _seed(self, seed=None):
    self.np_random, seed = seeding.np_random(seed)
    return [seed]

  def _sample_z(self, mu, logvar):
    z = mu + np.exp(logvar/2.0) * self.np_random.randn(*logvar.shape)
    return z

  def _reset(self):
    idx = self.np_random.randint(0, len(initial_mu_logvar))
    init_mu, init_logvar = initial_mu_logvar[idx]
    init_mu = np.array(init_mu)/10000.
    init_logvar = np.array(init_logvar)/10000.
    self.z = self._sample_z(init_mu, init_logvar)
    self.frame_count = 0
    return self.z

  def _sample_next_z(self, action):     
    s_model = self.rnn
    temperature = self.temperature

    sess = s_model.sess
    hps = s_model.hps
    
    OUTWIDTH = hps.output_seq_width

    prev_x = np.zeros((1, 1, OUTWIDTH))
    prev_x[0][0] = self.z

    strokes = np.zeros((1, OUTWIDTH), dtype=np.float32)

    input_x = np.concatenate((prev_x, action.reshape(1, 1, 3)), axis=2)
    feed = {s_model.input_x: input_x, s_model.initial_state:self.agent.state}
    [logmix, mean, logstd, self.agent.state] = sess.run([s_model.out_logmix, s_model.out_mean, s_model.out_logstd, s_model.final_state], feed)

    # adjust temperatures
    logmix2 = np.copy(logmix)/temperature
    logmix2 -= logmix2.max()
    logmix2 = np.exp(logmix2)
    logmix2 /= logmix2.sum(axis=1).reshape(OUTWIDTH, 1)

    mixture_idx = np.zeros(OUTWIDTH)
    chosen_mean = np.zeros(OUTWIDTH)
    chosen_logstd = np.zeros(OUTWIDTH)
    for j in range(OUTWIDTH):
      idx = get_pi_idx(self.np_random.rand(), logmix2[j])
      mixture_idx[j] = idx
      chosen_mean[j] = mean[j][idx]
      chosen_logstd[j] = logstd[j][idx]

    rand_gaussian = self.np_random.randn(OUTWIDTH)*np.sqrt(temperature)
    next_x = chosen_mean+np.exp(chosen_logstd)*rand_gaussian

    next_z = next_x.reshape(OUTWIDTH)

    return next_z

  def _step(self, action):
    self.frame_count += 1
    next_z = self._sample_next_z(action)
    reward = 0
    done = False
    if self.frame_count > 1200:
      done = True
    self.z = next_z
    return next_z, reward, done, {}

  def decode_obs(self, z):
    # decode the latent vector
    img = self.vae.decode(z.reshape(1, self.z_size)) * 255.
    img = np.round(img).astype(np.uint8)
    img = img.reshape(64, 64, 3)
    return img

  def _render(self, mode='human', close=False):

    img = self.decode_obs(self.z)

    img = resize(img, (int(np.round(SCREEN_Y*FACTOR)), int(np.round(SCREEN_X*FACTOR))))

    if self.frame_count > 0:
      pass
      #toimage(img, cmin=0, cmax=255).save('output/'+str(self.frame_count)+'.png')

    if close:
      if self.viewer is not None:
        self.viewer.close()
        self.viewer = None
      return

    if mode == 'rgb_array':
      return img

    elif mode == 'human':
      from gym.envs.classic_control import rendering
      if self.viewer is None:
        self.viewer = rendering.SimpleImageViewer()
      self.viewer.imshow(img)

def make_env(env_name, agent, seed=-1, render_mode=False):
  env = CarRacingDream(agent)
  if seed <0:
    seed = np.random.randint(2**31-1)
  env.seed(seed)

  return env
