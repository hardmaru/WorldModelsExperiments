# for the purpose of creating visualizations

import numpy as np
import random

import json
import sys

from dream_env import make_env
import time

from vae.vae import ConvVAE
from rnn.rnn import hps_sample, MDNRNN, rnn_init_state, rnn_next_state, rnn_output, rnn_output_size

render_mode = True

# controls whether we concatenate (z, c, h), etc for features used for car.
MODE_ZCH = 0
MODE_ZC = 1
MODE_Z = 2
MODE_Z_HIDDEN = 3 # extra hidden later
MODE_ZH = 4

EXP_MODE = MODE_ZH

def make_model(load_model=True):
  # can be extended in the future.
  model = Model(load_model=load_model)
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
  def __init__(self, load_model=True):
    self.env_name = "carracing"
    self.vae = ConvVAE(batch_size=1, gpu_mode=False, is_training=False, reuse=True)

    self.rnn = MDNRNN(hps_sample, gpu_mode=False, reuse=True)

    if load_model:
      self.vae.load_json('vae/vae.json')
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
    self.env = make_env(self.env_name, agent=self, seed=seed)

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
    #return np.random.randn(self.param_count)*stdev
    return np.random.standard_cauchy(self.param_count)*stdev # spice things up

  def init_random_model_params(self, stdev=0.1):
    params = self.get_random_model_params(stdev=stdev)
    self.set_model_params(params)

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

    z = model.env.reset()

    for t in range(max_episode_length):

      action = model.get_action(z)

      if render_mode:
        model.env.render("human")
      else:
        model.env.render('rgb_array')

      z, reward, done, info = model.env.step(action)

      if (render_mode):
        print("action", action, "sum.square.z", np.sum(np.square(z)))

      if done:
        break

    t_list.append(t)

  return reward_list, t_list

def main():

  assert len(sys.argv) > 0, 'python model.py path_to_model.json'

  use_model = False
  if (len(sys.argv) > 1):
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
    model.init_random_model_params(stdev=np.random.rand()*0.01)

  while(1):
    reward, steps_taken = simulate(model,
      train_mode=False, render_mode=render_mode, num_episode=1)
    print ("terminal reward", reward, "average steps taken", np.mean(steps_taken)+1)
    break

if __name__ == "__main__":
  main()
