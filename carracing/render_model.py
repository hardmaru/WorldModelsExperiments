# for the purpose of creating visualizations

import numpy as np
import random

import json
import sys

#from scipy.misc import imresize as resize, imsave

from render_env import make_env
import time

from vae.vae import ConvVAE
from rnn.rnn import hps_sample, MDNRNN, rnn_init_state, rnn_next_state, rnn_output, rnn_output_size

render_mode = False

# controls whether we concatenate (z, c, h), etc for features used for car.
MODE_ZCH = 0
MODE_ZC = 1
MODE_Z = 2
MODE_Z_HIDDEN = 3 # extra hidden later
MODE_ZH = 4

EXP_MODE = MODE_ZH

def make_model():
  # can be extended in the future.
  model = Model()
  return model

def sigmoid(x):
  return 1 / (1 + np.exp(-x))

def relu(x):
  return np.maximum(x, 0)

def clip(x, lo=0.0, hi=1.0):
  return np.minimum(np.maximum(x, lo), hi)

def passthru(x):
  return x

def softmax(x):
  e_x = np.exp(x - np.max(x))
  return e_x / e_x.sum(axis=0)

def sample(p):
  return np.argmax(np.random.multinomial(1, p))

class Model:
  ''' simple one layer model for car racing '''
  def __init__(self):
    self.env_name = "carracing"
    self.vae = ConvVAE(batch_size=1, gpu_mode=False, is_training=False, reuse=True)
    self.vae.load_json('vae/vae.json')
    self.rnn = MDNRNN(hps_sample, gpu_mode=False, reuse=True)
    self.rnn.load_json('rnn/rnn.json')
    self.state = rnn_init_state(self.rnn)
    self.rnn_mode = True

    self.input_size = rnn_output_size(EXP_MODE)
    self.z_size = 32

    if EXP_MODE == MODE_Z_HIDDEN: # one hidden layer
      self.hidden_size = 40
      self.weight_hidden = np.random.randn(self.input_size, self.hidden_size)
      self.bias_hidden = np.random.randn(self.hidden_size)
      self.weight_output = np.random.randn(self.hidden_size, 3)
      self.bias_output = np.random.randn(3)
      self.param_count = ((self.input_size+1)*self.hidden_size) + (self.hidden_size*3+3)
    else:
      self.weight = np.random.randn(self.input_size, 3)
      self.bias = np.random.randn(3)
      self.param_count = (self.input_size)*3+3

    self.render_mode = False

  def make_env(self, seed=-1, render_mode=False):
    self.render_mode = render_mode
    self.env = make_env(self.env_name, seed=seed, render_mode=render_mode)

  def reset(self):
    self.state = rnn_init_state(self.rnn)

  def encode_obs(self, obs):
    # convert raw obs to z, mu, logvar
    result = np.copy(obs).astype(np.float)/255.0
    result = result.reshape(1, 64, 64, 3)
    mu, logvar = self.vae.encode_mu_logvar(result)
    mu = mu[0]
    logvar = logvar[0]
    s = logvar.shape
    z = mu + np.exp(logvar/2.0) * np.random.randn(*s)
    return z, mu, logvar

  def decode_obs(self, z):
    # decode the latent vector
    img = self.vae.decode(z.reshape(1, self.z_size)) * 255.
    img = np.round(img).astype(np.uint8)
    img = img.reshape(64, 64, 3)
    return img

  def get_action(self, z):
    h = rnn_output(self.state, z, EXP_MODE)

    '''
    action = np.dot(h, self.weight) + self.bias
    action[0] = np.tanh(action[0])
    action[1] = sigmoid(action[1])
    action[2] = clip(np.tanh(action[2]))
    '''
    if EXP_MODE == MODE_Z_HIDDEN: # one hidden layer
      h = np.tanh(np.dot(h, self.weight_hidden) + self.bias_hidden)
      action = np.tanh(np.dot(h, self.weight_output) + self.bias_output)
    else:
      action = np.tanh(np.dot(h, self.weight) + self.bias)
    
    action[1] = (action[1]+1.0) / 2.0
    action[2] = clip(action[2])

    self.state = rnn_next_state(self.rnn, z, action, self.state)

    return action

  def set_model_params(self, model_params):
    if EXP_MODE == MODE_Z_HIDDEN: # one hidden layer
      params = np.array(model_params)
      cut_off = (self.input_size+1)*self.hidden_size
      params_1 = params[:cut_off]
      params_2 = params[cut_off:]
      self.bias_hidden = params_1[:self.hidden_size]
      self.weight_hidden = params_1[self.hidden_size:].reshape(self.input_size, self.hidden_size)
      self.bias_output = params_2[:3]
      self.weight_output = params_2[3:].reshape(self.hidden_size, 3)
    else:
      self.bias = np.array(model_params[:3])
      self.weight = np.array(model_params[3:]).reshape(self.input_size, 3)

  def load_model(self, filename):
    with open(filename) as f:    
      data = json.load(f)
    print('loading file %s' % (filename))
    self.data = data
    model_params = np.array(data[0]) # assuming other stuff is in data
    self.set_model_params(model_params)

  def get_random_model_params(self, stdev=0.1):
    return np.random.randn(self.param_count)*stdev

def simulate(model, train_mode=False, render_mode=True, num_episode=5, seed=-1, max_len=-1):

  reward_list = []
  t_list = []

  max_episode_length = 1000
  recording_mode = False
  penalize_turning = False

  if train_mode and max_len > 0:
    max_episode_length = max_len

  if (seed >= 0):
    random.seed(seed)
    np.random.seed(seed)
    model.env.seed(seed)

  for episode in range(num_episode):

    model.reset()

    obs = model.env.reset()

    total_reward = 0.0

    random_generated_int = np.random.randint(2**31-1)

    filename = "record/"+str(random_generated_int)+".npz"
    recording_mu = []
    recording_logvar = []
    recording_action = []
    recording_reward = [0]

    for t in range(max_episode_length):

      z, mu, logvar = model.encode_obs(obs)
      action = model.get_action(z)
      model.env.vae_frame = model.decode_obs(z)

      if render_mode:
        model.env.render("human")
      else:
        model.env.render('rgb_array')

        '''
        imsave("output/"+str(t)+".png", obs)
        # also save mu and sigma
        temp_mu = np.round(np.copy(mu*10000)).astype(np.int).tolist()
        temp_sigma = np.round(np.copy(np.exp(logvar/2.0)*10000)).astype(np.int).tolist()
        temp_out = [temp_mu, temp_sigma]
        with open("output/"+str(t)+".json", 'w') as f:
          json.dump(temp_out, f, separators=(',', ':'))
        '''

      recording_mu.append(mu)
      recording_logvar.append(logvar)
      recording_action.append(action)

      obs, reward, done, info = model.env.step(action)

      extra_reward = 0.0 # penalize for turning too frequently
      if train_mode and penalize_turning:
        extra_reward -= np.abs(action[0])/10.0
        reward += extra_reward

      recording_reward.append(reward)

      if (render_mode):
        print("action", action, "step reward", reward)

      total_reward += reward

      if done:
        break

    #for recording:
    z, mu, logvar = model.encode_obs(obs)
    action = model.get_action(z)

    recording_mu.append(mu)
    recording_logvar.append(logvar)
    recording_action.append(action)

    recording_mu = np.array(recording_mu, dtype=np.float16)
    recording_logvar = np.array(recording_logvar, dtype=np.float16)
    recording_action = np.array(recording_action, dtype=np.float16)
    recording_reward = np.array(recording_reward, dtype=np.float16)

    if not render_mode:
      if recording_mode:
        np.savez_compressed(filename, mu=recording_mu, logvar=recording_logvar, action=recording_action, reward=recording_reward)

    if render_mode:
      print("total reward", total_reward, "timesteps", t)
    reward_list.append(total_reward)
    t_list.append(t)

  return reward_list, t_list

def main():

  assert len(sys.argv) > 0, 'python model.py path_to_mode.json'

  use_model = False

  if len(sys.argv) > 1:
    use_model = True
    filename = sys.argv[1]
    print("filename", filename)

  the_seed = 0
  if len(sys.argv) > 2:
    the_seed = int(sys.argv[2])
    print("seed", the_seed)

  model = make_model()
  print('model size', model.param_count)

  model.make_env(render_mode=render_mode)

  if use_model:
    model.load_model(filename)
  else:
    params = model.get_random_model_params(stdev=0.1)
    model.set_model_params(params)

  while(1):
    reward, steps_taken = simulate(model,
      train_mode=False, render_mode=render_mode, num_episode=1)
    print ("terminal reward", reward, "average steps taken", np.mean(steps_taken)+1)
    break

if __name__ == "__main__":
  main()
