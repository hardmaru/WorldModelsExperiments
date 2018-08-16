import numpy as np
import matplotlib.pyplot as plt
import os
import tensorflow as tf
import random
from collections import namedtuple
import json
import time

import gym
from gym import spaces
from gym.utils import seeding

from scipy.misc import imresize as resize

np.set_printoptions(precision=4, edgeitems=6, linewidth=100, suppress=True)

model_path_name = 'tf_models'
model_rnn_size = 512
model_num_mixture = 5
model_restart_factor = 10.
model_state_space = 2 # includes C and H concatenated if 2, otherwise just H

TEMPERATURE = 1.25 # train with this temperature

# hyperparameters for our model. I was doing this on an older tf version, when HParams was not available ...

HyperParams = namedtuple('HyperParams', ['max_seq_len',
                                         'seq_width',
                                         'rnn_size',
                                         'batch_size',
                                         'grad_clip',
                                         'num_mixture',
                                         'restart_factor',
                                         'learning_rate',
                                         'decay_rate',
                                         'min_learning_rate',
                                         'use_layer_norm',
                                         'use_recurrent_dropout',
                                         'recurrent_dropout_prob',
                                         'use_input_dropout',
                                         'input_dropout_prob',
                                         'use_output_dropout',
                                         'output_dropout_prob',
                                         'is_training',
                                        ])

def default_hps():
  return HyperParams(max_seq_len=1000, # train on sequences of 1000
                     seq_width=64,    # width of our data (64)
                     rnn_size=model_rnn_size,    # number of rnn cells
                     batch_size=100,   # minibatch sizes
                     grad_clip=1.0,
                     num_mixture=model_num_mixture,   # number of mixtures in MDN
                     restart_factor=model_restart_factor, # factor of importance for restart=1 rare case for loss.
                     learning_rate=0.001,
                     decay_rate=0.99999,
                     min_learning_rate=0.00001,
                     use_layer_norm=0, # set this to 1 to get more stable results (less chance of NaN), but slower
                     use_recurrent_dropout=0,
                     recurrent_dropout_prob=0.90,
                     use_input_dropout=0,
                     input_dropout_prob=0.90,
                     use_output_dropout=0,
                     output_dropout_prob=0.90,
                     is_training=0)

hps_model = default_hps()
hps_sample = hps_model._replace(batch_size=1, max_seq_len=2, use_recurrent_dropout=0, is_training=0)

def reset_graph():
  if 'sess' in globals() and sess:
    sess.close()
  tf.reset_default_graph()

class ConvVAE(object):
  def __init__(self, z_size=64, batch_size=100, learning_rate=0.0001, kl_tolerance=0.5, is_training=True, reuse=False, gpu_mode=True):
    self.z_size = z_size
    self.batch_size = batch_size
    self.learning_rate = learning_rate
    self.is_training = is_training
    self.kl_tolerance = kl_tolerance
    self.reuse = reuse
    with tf.variable_scope('conv_vae', reuse=self.reuse):
      if not gpu_mode:
        with tf.device('/cpu:0'):
          tf.logging.info('Model using cpu.')
          self._build_graph()
      else:
        tf.logging.info('Model using gpu.')
        self._build_graph()
    self._init_session()
  def _build_graph(self):
    self.g = tf.Graph()
    with self.g.as_default():

      self.x = tf.placeholder(tf.float32, shape=[None, 64, 64, 3])

      # Encoder
      h = tf.layers.conv2d(self.x, 32, 4, strides=2, activation=tf.nn.relu, name="enc_conv1")
      h = tf.layers.conv2d(h, 64, 4, strides=2, activation=tf.nn.relu, name="enc_conv2")
      h = tf.layers.conv2d(h, 128, 4, strides=2, activation=tf.nn.relu, name="enc_conv3")
      h = tf.layers.conv2d(h, 256, 4, strides=2, activation=tf.nn.relu, name="enc_conv4")
      h = tf.reshape(h, [-1, 2*2*256])

      # VAE
      self.mu = tf.layers.dense(h, self.z_size, name="enc_fc_mu")
      self.logvar = tf.layers.dense(h, self.z_size, name="enc_fc_log_var")
      self.sigma = tf.exp(self.logvar / 2.0)
      self.epsilon = tf.random_normal([self.batch_size, self.z_size])
      self.z = self.mu + self.sigma * self.epsilon

      # Decoder
      h = tf.layers.dense(self.z, 4*256, name="dec_fc")
      h = tf.reshape(h, [-1, 1, 1, 4*256])
      h = tf.layers.conv2d_transpose(h, 128, 5, strides=2, activation=tf.nn.relu, name="dec_deconv1")
      h = tf.layers.conv2d_transpose(h, 64, 5, strides=2, activation=tf.nn.relu, name="dec_deconv2")
      h = tf.layers.conv2d_transpose(h, 32, 6, strides=2, activation=tf.nn.relu, name="dec_deconv3")
      self.y = tf.layers.conv2d_transpose(h, 3, 6, strides=2, activation=tf.nn.sigmoid, name="dec_deconv4")
      
      # train ops
      if self.is_training:
        self.global_step = tf.Variable(0, name='global_step', trainable=False)

        eps = 1e-6 # avoid taking log of zero
        
        # reconstruction loss (logistic), commented out.
        '''
        self.r_loss = - tf.reduce_mean(
          self.x * tf.log(self.y + eps) + (1.0 - self.x) * tf.log(1.0 - self.y + eps),
          reduction_indices = [1,2,3]
        )
        self.r_loss = tf.reduce_mean(self.r_loss)*64.0*64.0
        '''
        
        # reconstruction loss
        self.r_loss = tf.reduce_sum(
          tf.square(self.x - self.y),
          reduction_indices = [1,2,3]
        )
        self.r_loss = tf.reduce_mean(self.r_loss)

        # augmented kl loss per dim
        self.kl_loss = - 0.5 * tf.reduce_sum(
          (1 + self.logvar - tf.square(self.mu) - tf.exp(self.logvar)),
          reduction_indices = 1
        )
        self.kl_loss = tf.maximum(self.kl_loss, self.kl_tolerance * self.z_size)
        self.kl_loss = tf.reduce_mean(self.kl_loss)
        
        self.loss = self.r_loss + self.kl_loss
        
        # training
        self.lr = tf.Variable(self.learning_rate, trainable=False)
        self.optimizer = tf.train.AdamOptimizer(self.lr)
        grads = self.optimizer.compute_gradients(self.loss) # can potentially clip gradients here.

        self.train_op = self.optimizer.apply_gradients(
          grads, global_step=self.global_step, name='train_step')

      # initialize vars
      self.init = tf.global_variables_initializer()

  def _init_session(self):
    """Launch TensorFlow session and initialize variables"""
    self.sess = tf.Session(graph=self.g)
    self.sess.run(self.init)
  def close_sess(self):
    """ Close TensorFlow session """
    self.sess.close()
  def encode(self, x):
    return self.sess.run(self.z, feed_dict={self.x: x})
  def encode_mu_logvar(self, x):
    (mu, logvar) = self.sess.run([self.mu, self.logvar], feed_dict={self.x: x})
    return mu, logvar
  def decode(self, z):
    return self.sess.run(self.y, feed_dict={self.z: z})
  def get_model_params(self):
    # get trainable params.
    model_names = []
    model_params = []
    model_shapes = []
    with self.g.as_default():
      t_vars = tf.trainable_variables()
      for var in t_vars:
        param_name = var.name
        p = self.sess.run(var)
        model_names.append(param_name)
        params = np.round(p*10000).astype(np.int).tolist()
        model_params.append(params)
        model_shapes.append(p.shape)
    return model_params, model_shapes, model_names
  def get_random_model_params(self, stdev=0.5):
    # get random params.
    _, mshape, _ = self.get_model_params()
    rparam = []
    for s in mshape:
      #rparam.append(np.random.randn(*s)*stdev)
      rparam.append(np.random.standard_cauchy(s)*stdev) # spice things up!
    return rparam
  def set_model_params(self, params):
    with self.g.as_default():
      t_vars = tf.trainable_variables()
      idx = 0
      for var in t_vars:
        pshape = self.sess.run(var).shape
        p = np.array(params[idx])
        assert pshape == p.shape, "inconsistent shape"
        assign_op = var.assign(p.astype(np.float)/10000.)
        self.sess.run(assign_op)
        idx += 1
  def load_json(self, jsonfile='vae.json'):
    with open(jsonfile, 'r') as f:
      params = json.load(f)
    self.set_model_params(params)
  def save_json(self, jsonfile='vae.json'):
    model_params, model_shapes, model_names = self.get_model_params()
    qparams = []
    for p in model_params:
      qparams.append(p)
    with open(jsonfile, 'wt') as outfile:
      json.dump(qparams, outfile, sort_keys=True, indent=0, separators=(',', ': '))
  def set_random_params(self, stdev=0.5):
    rparam = self.get_random_model_params(stdev)
    self.set_model_params(rparam)
  def save_model(self, model_save_path):
    sess = self.sess
    with self.g.as_default():
      saver = tf.train.Saver(tf.global_variables())
    checkpoint_path = os.path.join(model_save_path, 'vae')
    tf.logging.info('saving model %s.', checkpoint_path)
    saver.save(sess, checkpoint_path, 0) # just keep one
  def load_checkpoint(self, checkpoint_path):
    sess = self.sess
    with self.g.as_default():
      saver = tf.train.Saver(tf.global_variables())
    ckpt = tf.train.get_checkpoint_state(checkpoint_path)
    print('loading model', ckpt.model_checkpoint_path)
    tf.logging.info('Loading model %s.', ckpt.model_checkpoint_path)
    saver.restore(sess, ckpt.model_checkpoint_path)

# MDN-RNN model tailored for doomrnn
class Model():
  def __init__(self, hps, gpu_mode=True, reuse=False):
    self.hps = hps
    with tf.variable_scope('mdn_rnn', reuse=reuse):
      if not gpu_mode:
        with tf.device("/cpu:0"):
          print("model using cpu")
          self.g = tf.Graph()
          with self.g.as_default():
            self.build_model(hps)
      else:
        print("model using gpu")
        self.g = tf.Graph()
        with self.g.as_default():
          self.build_model(hps)
    self.init_session()
  def build_model(self, hps):
    
    self.num_mixture = hps.num_mixture
    KMIX = self.num_mixture # 5 mixtures
    WIDTH = hps.seq_width # 64 channels
    LENGTH = self.hps.max_seq_len-1 # 999 timesteps

    if hps.is_training:
      self.global_step = tf.Variable(0, name='global_step', trainable=False)

    cell_fn = tf.contrib.rnn.LayerNormBasicLSTMCell # use LayerNormLSTM

    use_recurrent_dropout = False if self.hps.use_recurrent_dropout == 0 else True
    use_input_dropout = False if self.hps.use_input_dropout == 0 else True
    use_output_dropout = False if self.hps.use_output_dropout == 0 else True
    is_training = False if self.hps.is_training == 0 else True
    use_layer_norm = False if self.hps.use_layer_norm == 0 else True

    if use_recurrent_dropout:
      cell = cell_fn(hps.rnn_size, layer_norm=use_layer_norm, dropout_keep_prob=self.hps.recurrent_dropout_prob)
    else:
      cell = cell_fn(hps.rnn_size, layer_norm=use_layer_norm)

    # multi-layer, and dropout:
    print("input dropout mode =", use_input_dropout)
    print("output dropout mode =", use_output_dropout)
    print("recurrent dropout mode =", use_recurrent_dropout)
    if use_input_dropout:
      print("applying dropout to input with keep_prob =", self.hps.input_dropout_prob)
      cell = tf.contrib.rnn.DropoutWrapper(cell, input_keep_prob=self.hps.input_dropout_prob)
    if use_output_dropout:
      print("applying dropout to output with keep_prob =", self.hps.output_dropout_prob)
      cell = tf.contrib.rnn.DropoutWrapper(cell, output_keep_prob=self.hps.output_dropout_prob)
    self.cell = cell

    self.sequence_lengths = LENGTH # assume every sample has same length.
    self.batch_z = tf.placeholder(dtype=tf.float32, shape=[self.hps.batch_size, self.hps.max_seq_len, WIDTH])
    self.batch_action = tf.placeholder(dtype=tf.float32, shape=[self.hps.batch_size, self.hps.max_seq_len])
    self.batch_restart = tf.placeholder(dtype=tf.float32, shape=[self.hps.batch_size, self.hps.max_seq_len])

    self.input_z = self.batch_z[:, :LENGTH, :]
    self.input_action = self.batch_action[:, :LENGTH]
    self.input_restart = self.batch_restart[: , :LENGTH]

    self.target_z = self.batch_z[:, 1:, :]
    self.target_restart = self.batch_restart[: , 1:]
    
    self.input_seq = tf.concat([self.input_z,
                                tf.reshape(self.input_action, [self.hps.batch_size, LENGTH, 1]),
                                tf.reshape(self.input_restart, [self.hps.batch_size, LENGTH, 1])], axis=2)

    self.zero_state = cell.zero_state(batch_size=hps.batch_size, dtype=tf.float32)
    self.initial_state = self.zero_state

    inputs = tf.unstack(self.input_seq, axis=1)

    def custom_rnn_autodecoder(decoder_inputs, input_restart, initial_state, cell, scope=None):
      # customized rnn_decoder for the task of dealing with restart
      with tf.variable_scope(scope or "RNN"):
        state = initial_state
        zero_c, zero_h = self.zero_state
        outputs = []
        prev = None

        for i in range(LENGTH):
          inp = decoder_inputs[i]
          if i > 0:
            tf.get_variable_scope().reuse_variables()

          # if restart is 1, then set lstm state to zero
          restart_flag = tf.greater(input_restart[:, i], 0.5)

          c, h = state

          c = tf.where(restart_flag, zero_c, c)
          h = tf.where(restart_flag, zero_h, h)

          output, state = cell(inp, tf.nn.rnn_cell.LSTMStateTuple(c, h))
          outputs.append(output)

      return outputs, state

    outputs, final_state = custom_rnn_autodecoder(inputs, self.input_restart, self.initial_state, self.cell)
    output = tf.reshape(tf.concat(outputs, axis=1), [-1, self.hps.rnn_size])

    NOUT = WIDTH * KMIX * 3 + 1 # plus 1 to predict the restart state.

    with tf.variable_scope('RNN'):
      output_w = tf.get_variable("output_w", [self.hps.rnn_size, NOUT])
      output_b = tf.get_variable("output_b", [NOUT])

    output = tf.reshape(output, [-1, hps.rnn_size])
    output = tf.nn.xw_plus_b(output, output_w, output_b)
    
    self.out_restart_logits = output[:, 0]
    output = output[:, 1:]
    
    output = tf.reshape(output, [-1, KMIX * 3])
    self.final_state = final_state    

    logSqrtTwoPI = np.log(np.sqrt(2.0 * np.pi))

    def tf_lognormal(y, mean, logstd):
      return -0.5 * ((y - mean) / tf.exp(logstd)) ** 2 - logstd - logSqrtTwoPI

    def get_lossfunc(logmix, mean, logstd, y):
      v = logmix + tf_lognormal(y, mean, logstd)
      v = tf.reduce_logsumexp(v, 1, keepdims=True)
      return -tf.reduce_mean(v)

    def get_mdn_coef(output):
      logmix, mean, logstd = tf.split(output, 3, 1)
      logmix = logmix - tf.reduce_logsumexp(logmix, 1, keepdims=True)
      return logmix, mean, logstd

    out_logmix, out_mean, out_logstd = get_mdn_coef(output)

    self.out_logmix = out_logmix
    self.out_mean = out_mean
    self.out_logstd = out_logstd

    # reshape target data so that it is compatible with prediction shape
    flat_target_data = tf.reshape(self.target_z,[-1, 1])

    lossfunc = get_lossfunc(out_logmix, out_mean, out_logstd, flat_target_data)

    self.z_cost = tf.reduce_mean(lossfunc)

    flat_target_restart = tf.reshape(self.target_restart, [-1, 1])

    self.r_cost = tf.nn.sigmoid_cross_entropy_with_logits(labels=flat_target_restart,
                                                          logits=tf.reshape(self.out_restart_logits,[-1, 1]))
    
    factor = tf.ones_like(self.r_cost) + flat_target_restart * (self.hps.restart_factor-1.0)

    self.r_cost = tf.reduce_mean(tf.multiply(factor, self.r_cost))

    self.cost = self.z_cost + self.r_cost

    if self.hps.is_training == 1:
      self.lr = tf.Variable(self.hps.learning_rate, trainable=False)
      optimizer = tf.train.AdamOptimizer(self.lr)

      gvs = optimizer.compute_gradients(self.cost)
      capped_gvs = [(tf.clip_by_value(grad, -self.hps.grad_clip, self.hps.grad_clip), var) for grad, var in gvs]
      self.train_op = optimizer.apply_gradients(capped_gvs, global_step=self.global_step, name='train_step')

    # initialize vars
    self.init = tf.global_variables_initializer()
  def init_session(self):
    """Launch TensorFlow session and initialize variables"""
    self.sess = tf.Session(graph=self.g)
    self.sess.run(self.init)
  def close_sess(self):
    """ Close TensorFlow session """
    self.sess.close()
  def save_model(self, model_save_path, epoch):
    sess = self.sess
    with self.g.as_default():
      saver = tf.train.Saver(tf.global_variables())
    checkpoint_path = os.path.join(model_save_path, 'doomcover_rnn')
    tf.logging.info('saving model %s.', checkpoint_path)
    saver.save(sess, checkpoint_path, epoch) # just keep one
  def load_checkpoint(self, checkpoint_path):
    sess = self.sess
    with self.g.as_default():
      saver = tf.train.Saver(tf.global_variables())
    ckpt = tf.train.get_checkpoint_state(checkpoint_path)
    print('loading model', ckpt.model_checkpoint_path)
    tf.logging.info('Loading model %s.', ckpt.model_checkpoint_path)
    saver.restore(sess, ckpt.model_checkpoint_path)
  def get_model_params(self):
    # get trainable params.
    model_names = []
    model_params = []
    model_shapes = []
    with self.g.as_default():
      t_vars = tf.trainable_variables()
      for var in t_vars:
        param_name = var.name
        p = self.sess.run(var)
        model_names.append(param_name)
        params = np.round(p*10000).astype(np.int).tolist()
        model_params.append(params)
        model_shapes.append(p.shape)
    return model_params, model_shapes, model_names
  def set_model_params(self, params):
    with self.g.as_default():
      t_vars = tf.trainable_variables()
      idx = 0
      for var in t_vars:
        pshape = self.sess.run(var).shape
        p = np.array(params[idx])
        assert pshape == p.shape, "inconsistent shape"
        assign_op = var.assign(p.astype(np.float)/10000.)
        self.sess.run(assign_op)
        idx += 1
  def get_random_model_params(self, stdev=0.5):
    # get random params.
    _, mshape, _ = self.get_model_params()
    rparam = []
    for s in mshape:
      #rparam.append(np.random.randn(*s)*stdev)
      rparam.append(np.random.standard_cauchy(s)*stdev) # spice things up!
    return rparam
  def set_random_params(self, stdev=0.5):
    rparam = self.get_random_model_params(stdev)
    self.set_model_params(rparam)
  def load_json(self, jsonfile='rnn.json'):
    with open(jsonfile, 'r') as f:
      params = json.load(f)
    self.set_model_params(params)
  def save_json(self, jsonfile='rnn.json'):
    model_params, model_shapes, model_names = self.get_model_params()
    qparams = []
    for p in model_params:
      qparams.append(p)
    with open(jsonfile, 'wt') as outfile:
      json.dump(qparams, outfile, sort_keys=True, indent=0, separators=(',', ': '))

def get_pi_idx(x, pdf):
  # samples from a categorial distribution
  N = pdf.size
  accumulate = 0
  for i in range(0, N):
    accumulate += pdf[i]
    if (accumulate >= x):
      return i
  random_value = np.random.randint(N)
  #print('error with sampling ensemble, returning random', random_value)
  return random_value

def sigmoid(x):
  return 1 / (1 + np.exp(-x))

class DoomCoverRNNEnv(gym.Env):
  metadata = {
    'render.modes': ['human', 'rgb_array'],
    'video.frames_per_second' : 50
  }

  def __init__(self, render_mode=False, load_model=True):

    self.render_mode = render_mode

    with open(os.path.join(model_path_name, 'initial_z.json'), 'r') as f:
      [initial_mu, initial_logvar] = json.load(f)

    self.initial_mu_logvar = [list(elem) for elem in zip(initial_mu, initial_logvar)]

    reset_graph()

    self.vae = ConvVAE(batch_size=1, gpu_mode=False, is_training=False, reuse=True)

    self.rnn = Model(hps_sample, gpu_mode=False)

    if load_model:
      self.vae.load_json(os.path.join(model_path_name, 'vae.json'))
      self.rnn.load_json(os.path.join(model_path_name, 'rnn.json'))

    # future versions of OpenAI gym needs a dtype=np.float32 in the next line:
    self.action_space = spaces.Box(low=-1.0, high=1.0, shape=())
    self.outwidth = self.rnn.hps.seq_width
    self.obs_size = self.outwidth + model_rnn_size * model_state_space
    # future versions of OpenAI gym needs a dtype=np.float32 in the next line:
    self.observation_space = spaces.Box(low=-50., high=50., shape=(self.obs_size,))

    self.zero_state = self.rnn.sess.run(self.rnn.zero_state)

    self.seed()

    self.rnn_state = None
    self.z = None
    self.restart = None
    self.temperature = None
    
    self.frame_count = None
    self.max_frame = 2100

    self.viewer = None
    
    self.reset()

  def _sample_init_z(self):
    idx = self.np_random.randint(0, len(self.initial_mu_logvar))
    init_mu, init_logvar = self.initial_mu_logvar[idx]
    init_mu = np.array(init_mu)/10000.
    init_logvar = np.array(init_logvar)/10000.
    init_z = init_mu + np.exp(init_logvar/2.0) * self.np_random.randn(*init_logvar.shape)
    return init_z

  def _current_state(self):
    if model_state_space == 2:
      return np.concatenate([self.z, self.rnn_state.c.flatten(), self.rnn_state.h.flatten()], axis=0)
    return np.concatenate([self.z, self.rnn_state.h.flatten()], axis=0)

  def _reset(self):
    self.temperature = TEMPERATURE
    self.rnn_state = self.zero_state
    self.z = self._sample_init_z()
    self.restart = 1
    self.frame_count = 0
    return self._current_state()

  def _seed(self, seed=None):
    if seed:
      tf.set_random_seed(seed)
    self.np_random, seed = seeding.np_random(seed)
    return [seed]

  def _step(self, action):
    
    self.frame_count += 1

    prev_z = np.zeros((1, 1, self.outwidth))
    prev_z[0][0] = self.z

    prev_action = np.zeros((1, 1))
    prev_action[0] = action
    
    prev_restart = np.ones((1, 1))
    prev_restart[0] = self.restart
    
    s_model = self.rnn
    temperature = self.temperature

    feed = {s_model.input_z: prev_z,
            s_model.input_action: prev_action,
            s_model.input_restart: prev_restart,
            s_model.initial_state: self.rnn_state
           }

    [logmix, mean, logstd, logrestart, next_state] = s_model.sess.run([s_model.out_logmix,
                                                                       s_model.out_mean,
                                                                       s_model.out_logstd,
                                                                       s_model.out_restart_logits,
                                                                       s_model.final_state],
                                                                      feed)
    
    OUTWIDTH = self.outwidth

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
    next_z = chosen_mean+np.exp(chosen_logstd)*rand_gaussian

    next_restart = 0
    done = False
    if (logrestart[0] > 0):
      next_restart = 1
      done = True
    
    self.z = next_z
    self.restart = next_restart
    self.rnn_state = next_state
    
    reward = 1 # always return a reward of one if still alive.
    
    if self.frame_count >= self.max_frame:
      done = True

    return self._current_state(), reward, done, {}

  def _get_image(self, upsize=False):
    # decode the latent vector
    img = self.vae.decode(self.z.reshape(1, 64)) * 255.
    img = np.round(img).astype(np.uint8)
    img = img.reshape(64, 64, 3)
    if upsize:
      img = resize(img, (640, 640))
    return img

  def _render(self, mode='human', close=False):
    if not self.render_mode:
      return

    if close:
      if self.viewer is not None:
        self.viewer.close()
        self.viewer = None
      return

    if mode == 'rgb_array':
      img = self._get_image(upsize=True)
      return img

    elif mode == 'human':
      img = self._get_image(upsize=True)
      from gym.envs.classic_control import rendering
      if self.viewer is None:
        self.viewer = rendering.SimpleImageViewer()
      self.viewer.imshow(img)

if __name__=="__main__":

  env = DoomCoverRNNEnv(render_mode=True)

  if env.render_mode:
    from pyglet.window import key
  a = np.array( [0.0] )
  overwrite = False

  def key_press(k, mod):
    global overwrite
    overwrite = True
    if k==key.LEFT:
      a[0] = -1.0
      print('human key left.')
    if k==key.RIGHT:
      a[0] = +1.0
      print('human key right.')

  def key_release(k, mod):
    a[0] = 0.

  if env.render_mode:
    env.render()
    env.viewer.window.on_key_press = key_press
    env.viewer.window.on_key_release = key_release

  reward_list = []

  for i in range(40):
    env.reset()
    total_reward = 0.0
    steps = 0

    repeat = np.random.randint(1, 11)
    obs_list = []
    z_list = []
    obs = env.reset()
    obs_list.append(obs)
    z_list.append(obs[0:64])

    overwrite = False

    while True:
      if steps % repeat == 0:
        action = np.random.rand() * 2.0 - 1.0
        repeat = np.random.randint(1, 11)
      action = 0.0
      if overwrite:
        action = a[0]
      obs, reward, done, info = env.step(action)
      obs_list.append(obs)
      z_list.append(obs[0:64])
      total_reward += reward
      steps += 1

      if env.render_mode:
        env.render()
      if done:
        break

    reward_list.append(total_reward)

    print('cumulative reward', total_reward)
  print('average reward', np.mean(reward_list))
