import numpy as np
import random
#from scipy.fftpack import dct
import json
import sys
import config
from env import make_env
import time

final_mode = True
render_mode = True
RENDER_DELAY = False

def make_model(game):
  # can be extended in the future.
  model = Model(game)
  return model

def sigmoid(x):
  return 1 / (1 + np.exp(-x))

def relu(x):
  return np.maximum(x, 0)

def passthru(x):
  return x

def softmax(x):
  e_x = np.exp(x - np.max(x))
  return e_x / e_x.sum(axis=0)

def sample(p):
  return np.argmax(np.random.multinomial(1, p))

class Model:
  ''' simple feedforward model '''
  def __init__(self, game):
    self.noise_level = 0.0
    self.env_name = game.env_name

    self.input_size = game.input_size
    self.output_size = game.output_size

    self.shapes = [ (game.input_size, game.output_size) ]

    self.sample_output = False
    if game.activation == 'relu':
      self.activations = [relu]
    elif game.activation == 'sigmoid':
      self.activations = [sigmoid]
    elif game.activation == 'passthru':
      self.activations = [np.tanh]
    else:
      self.activations = [np.tanh]

    self.weight = []
    self.param_count = game.input_size * game.output_size

    for shape in self.shapes:
      self.weight.append(np.zeros(shape=shape))

    self.render_mode = False

  def make_env(self, seed=-1, render_mode=False, load_model=True):
    self.render_mode = render_mode
    self.env = make_env(self.env_name, seed=seed, render_mode=render_mode, load_model=load_model)

  def get_action(self, x):
    # if mean_mode = True, ignore sampling.
    h = np.array(x).flatten()

    num_layers = len(self.weight)
    for i in range(num_layers):
      w = self.weight[i]
      h = np.matmul(h, w)
      h = self.activations[i](h + np.random.randn()*self.noise_level)

    return h

  def set_model_params(self, model_params):
    pointer = 0
    for i in range(len(self.shapes)):
      w_shape = self.shapes[i]
      s_w = np.product(w_shape)
      s = s_w
      chunk = np.array(model_params[pointer:pointer+s])
      self.weight[i] = chunk[:s_w].reshape(w_shape)
      pointer += s

  def load_model(self, filename):
    with open(filename) as f:    
      data = json.load(f)
    print('loading file %s' % (filename))
    self.data = data
    model_params = np.array(data[0]) # assuming other stuff is in data
    self.set_model_params(model_params)
    # also load the vae and rnn
    self.env.vae.load_json('tf_models/vae.json')
    self.env.rnn.load_json('tf_models/rnn.json')

  def get_random_model_params(self, stdev=0.1):
    #return np.random.randn(self.param_count)*stdev
    return np.random.standard_cauchy(self.param_count)*stdev # spice things up!

  def init_random_model_params(self, stdev=0.1):
    params = self.get_random_model_params(stdev=stdev)
    self.set_model_params(params)
    vae_params = self.env.vae.get_random_model_params(stdev=stdev)
    self.env.vae.set_model_params(vae_params)
    rnn_params = self.env.rnn.get_random_model_params(stdev=stdev)
    self.env.rnn.set_model_params(rnn_params)

def evaluate(model):
  # run 100 times and average score, according to the reles.
  model.env.seed(0)
  total_reward = 0.0
  N = 100
  for i in range(N):
    reward, t = simulate(model, train_mode=False, render_mode=False, num_episode=1)
    total_reward += reward[0]
  return (total_reward / float(N))

def simulate(model, train_mode=False, render_mode=True, num_episode=5, seed=-1, max_len=-1):

  reward_list = []
  t_list = []

  max_episode_length = 2100

  if train_mode and max_len > 0:
    max_episode_length = max_len

  if (seed >= 0):
    random.seed(seed)
    np.random.seed(seed)
    model.env.seed(seed)

  for episode in range(num_episode):

    obs = model.env.reset()

    if obs is None:
      obs = np.zeros(model.input_size)

    total_reward = 0.0

    for t in range(max_episode_length):

      if render_mode:
        model.env.render("human")
        if RENDER_DELAY:
          time.sleep(0.01)

      action = model.get_action(obs)

      prev_obs = obs

      obs, reward, done, info = model.env.step(action)

      if (render_mode):
        pass
        #print("action", action, "step reward", reward)
        #print("step reward", reward)
      total_reward += reward

      if done:
        break

    if render_mode:
      print("reward", total_reward, "timesteps", t)
      model.env.close()

    reward_list.append(total_reward)
    t_list.append(t)

  return reward_list, t_list

def main():

  global RENDER_DELAY
  global final_mode

  assert len(sys.argv) > 2, 'python model.py gamename render/norender path_to_model.json [seed]'

  gamename = sys.argv[1]

  game = config.games[gamename]

  final_mode_string = str(sys.argv[2])
  if (final_mode_string == "render"):
    final_mode = False # don't run 100 times, just visualize results.

  use_model = False
  if (len(sys.argv) > 3):
    use_model = True
    filename = sys.argv[3]
    print("filename", filename)

  the_seed = np.random.randint(10000)
  if len(sys.argv) > 4:
    the_seed = int(sys.argv[4])
    print("seed", the_seed)

  model = make_model(game)
  print('model size', model.param_count)


  if (use_model):
    model.make_env(render_mode=render_mode)
    model.load_model(filename)
  else:
    model.make_env(render_mode=render_mode, load_model=False)
    model.init_random_model_params(stdev=np.random.rand()*0.01)

  if final_mode:
    total_reward = 0.0
    np.random.seed(the_seed)
    model.env.seed(the_seed)
    reward_list = []

    for i in range(100):
      reward, steps_taken = simulate(model, train_mode=False, render_mode=False, num_episode=1)
      print("iteration", i, "reward", reward[0])
      total_reward += reward[0]
      reward_list.append(reward[0])
    print("seed", the_seed, "average_reward", total_reward/100, "stdev", np.std(reward_list))

  else:

    reward, steps_taken = simulate(model,
      train_mode=False, render_mode=render_mode, num_episode=1)
    print ("terminal reward", reward, "average steps taken", np.mean(steps_taken)+1)

if __name__ == "__main__":
  main()
