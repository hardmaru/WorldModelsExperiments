'''
train mdn-rnn from pre-processed data.
also save 1000 initial mu and logvar, for generative experiments (not related to training).
'''

import numpy as np
import os
import json
import tensorflow as tf
import random
import time

from doomrnn import reset_graph, HyperParams, Model

os.environ["CUDA_VISIBLE_DEVICES"]="0"
np.set_printoptions(precision=4, edgeitems=6, linewidth=100, suppress=True)

model_save_path = "tf_rnn"
model_rnn_size = 512
model_num_mixture = 5
model_restart_factor = 10.

DATA_DIR = "series"

if not os.path.exists(model_save_path):
  os.makedirs(model_save_path)
  
initial_z_save_path = "tf_initial_z"
if not os.path.exists(initial_z_save_path):
  os.makedirs(initial_z_save_path)

def default_hps():
  return HyperParams(max_seq_len=500, # train on sequences of 500 (found it worked better than 1000)
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
                     is_training=1)

hps_model = default_hps()
hps_sample = hps_model._replace(batch_size=1, max_seq_len=2, use_recurrent_dropout=0, is_training=0)

# load preprocessed data
raw_data = np.load(os.path.join(DATA_DIR, "series.npz"))
raw_data_mu = raw_data["mu"]
raw_data_logvar = raw_data["logvar"]
raw_data_action =  raw_data["action"]

def load_series_data():
  all_data = []
  for i in range(len(raw_data_mu)):
    action = raw_data_action[i]
    mu = raw_data_mu[i]
    logvar = raw_data_logvar[i]
    all_data.append([mu, logvar, action])
  return all_data

def get_frame_count(all_data):
  frame_count = []
  for data in all_data:
    frame_count.append(len(data[0]))
  return np.sum(frame_count)

def create_batches(all_data, batch_size=100, seq_length=500):
  num_frames = get_frame_count(all_data)
  num_batches = int(num_frames/(batch_size*seq_length))
  num_frames_adjusted = num_batches*batch_size*seq_length
  random.shuffle(all_data)
  num_frames = get_frame_count(all_data)
  data_mu = np.zeros((num_frames, N_z), dtype=np.float16)
  data_logvar = np.zeros((num_frames, N_z), dtype=np.float16)
  data_action = np.zeros(num_frames, dtype=np.float16)
  data_restart = np.zeros(num_frames, dtype=np.uint8)
  idx = 0
  for data in all_data:
    mu, logvar, action=data
    N = len(action)
    data_mu[idx:idx+N] = mu.reshape(N, 64)
    data_logvar[idx:idx+N] = logvar.reshape(N, 64)
    data_action[idx:idx+N] = action.reshape(N)
    data_restart[idx]=1
    idx += N

  data_mu = data_mu[0:num_frames_adjusted]
  data_logvar = data_logvar[0:num_frames_adjusted]
  data_action = data_action[0:num_frames_adjusted]
  data_restart = data_restart[0:num_frames_adjusted]

  data_mu = np.split(data_mu.reshape(batch_size, -1, 64), num_batches, 1)
  data_logvar = np.split(data_logvar.reshape(batch_size, -1, 64), num_batches, 1)
  data_action = np.split(data_action.reshape(batch_size, -1), num_batches, 1)
  data_restart = np.split(data_restart.reshape(batch_size, -1), num_batches, 1)

  return data_mu, data_logvar, data_action, data_restart

def get_batch(batch_idx, data_mu, data_logvar, data_action, data_restart):
  batch_mu = data_mu[batch_idx]
  batch_logvar = data_logvar[batch_idx]
  batch_action = data_action[batch_idx]
  batch_restart = data_restart[batch_idx]
  batch_s = batch_logvar.shape
  batch_z = batch_mu + np.exp(batch_logvar/2.0) * np.random.randn(*batch_s)
  return batch_z, batch_action, batch_restart

# process data
all_data = load_series_data()

max_seq_len = hps_model.max_seq_len
N_z = hps_model.seq_width

# save 1000 initial mu and logvars:
initial_mu = []
initial_logvar = []
for i in range(1000):
  mu = np.copy(raw_data_mu[i][0, :]*10000).astype(np.int).tolist()
  logvar = np.copy(raw_data_logvar[i][0, :]*10000).astype(np.int).tolist()
  initial_mu.append(mu)
  initial_logvar.append(logvar)
with open(os.path.join("tf_initial_z", "initial_z.json"), 'wt') as outfile:
  json.dump([initial_mu, initial_logvar], outfile, sort_keys=True, indent=0, separators=(',', ': '))

reset_graph()
model = Model(hps_model)

hps = hps_model
start = time.time()

for epoch in range(1, 401):
  print('preparing data for epoch', epoch)
  data_mu, data_logvar, data_action, data_restart = 0, 0, 0, 0
  data_mu, data_logvar, data_action, data_restart = create_batches(all_data)
  num_batches = len(data_mu)
  print('number of batches', num_batches)
  end = time.time()
  time_taken = end-start
  print('time taken to create batches', time_taken)

  batch_state = model.sess.run(model.initial_state)

  for local_step in range(num_batches):
  
    batch_z, batch_action, batch_restart = get_batch(local_step, data_mu, data_logvar, data_action, data_restart)
    step = model.sess.run(model.global_step)
    curr_learning_rate = (hps.learning_rate-hps.min_learning_rate) * (hps.decay_rate) ** step + hps.min_learning_rate

    feed = {model.batch_z: batch_z,
            model.batch_action: batch_action,
            model.batch_restart: batch_restart,
            model.initial_state: batch_state,
            model.lr: curr_learning_rate}

    (train_cost, z_cost, r_cost, batch_state, train_step, _) = model.sess.run([model.cost, model.z_cost, model.r_cost, model.final_state, model.global_step, model.train_op], feed)
    if (step%20==0 and step > 0):
      end = time.time()
      time_taken = end-start
      start = time.time()
      output_log = "step: %d, lr: %.6f, cost: %.4f, z_cost: %.4f, r_cost: %.4f, train_time_taken: %.4f" % (step, curr_learning_rate, train_cost, z_cost, r_cost, time_taken)
      print(output_log)

# save the model (don't bother with tf checkpoints json all the way ...)
model.save_json(os.path.join(model_save_path, "rnn.json"))

