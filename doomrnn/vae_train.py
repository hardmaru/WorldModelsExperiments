'''
Train VAE model on data created using extract.py
final model saved into tf_vae/vae.json
'''

import os
os.environ["CUDA_VISIBLE_DEVICES"]="0" # can just override for multi-gpu systems

import tensorflow as tf
import random
import numpy as np
np.set_printoptions(precision=4, edgeitems=6, linewidth=100, suppress=True)

from doomrnn import reset_graph, ConvVAE

# Hyperparameters for ConvVAE
z_size=64
batch_size=100
learning_rate=0.0001
kl_tolerance=0.5

# Parameters for training
NUM_EPOCH = 10
DATA_DIR = "record"

model_save_path = "tf_vae"
if not os.path.exists(model_save_path):
  os.makedirs(model_save_path)

def load_raw_data_list(filelist):
  data_list = []
  counter = 0
  for i in range(len(filelist)):
    filename = filelist[i]
    raw_data = np.load(os.path.join(DATA_DIR, filename))['obs']
    data_list.append(raw_data)
    if ((i+1) % 1000 == 0):
      print("loading file", (i+1))
  return data_list

def count_length_of_raw_data(raw_data_list):
  min_len = 100000
  max_len = 0
  N = len(raw_data_list)
  total_length = 0
  for i in range(N):
    l = len(raw_data_list[i])
    if l > max_len:
      max_len = l
    if l < min_len:
      min_len = l
    if l < 10:
      print(i)
    total_length += l
  return  total_length

def create_dataset(raw_data_list):
  N = len(raw_data_list)
  M = count_length_of_raw_data(raw_data_list)
  data = np.zeros((M, 64, 64, 3), dtype=np.uint8)
  idx = 0
  for i in range(N):
    raw_data = raw_data_list[i]
    l = len(raw_data)
    if (idx+l) > M:
      data = data[0:idx]
      break
    data[idx:idx+l] = raw_data
    idx += l
  return data

# load dataset from record/*. only use first 10K, sorted by filename.
filelist = os.listdir(DATA_DIR)
filelist.sort()
filelist = filelist[0:10000]
dataset = load_raw_data_list(filelist)
dataset = create_dataset(dataset)

# split into batches:
total_length = len(dataset)
num_batches = int(np.floor(total_length/batch_size))
print("num_batches", num_batches)

reset_graph()

vae = ConvVAE(z_size=z_size,
              batch_size=batch_size,
              learning_rate=learning_rate,
              kl_tolerance=kl_tolerance,
              is_training=True,
              reuse=False,
              gpu_mode=True)

# train loop:
print("train", "step", "loss", "recon_loss", "kl_loss")
for epoch in range(NUM_EPOCH):
  np.random.shuffle(dataset)
  for idx in range(num_batches):
    batch = dataset[idx*batch_size:(idx+1)*batch_size]

    obs = batch.astype(np.float)/255.0

    feed = {vae.x: obs,}

    (train_loss, r_loss, kl_loss, train_step, _) = vae.sess.run([
      vae.loss, vae.r_loss, vae.kl_loss, vae.global_step, vae.train_op
    ], feed)
  
    if ((train_step+1) % 500 == 0):
      print("step", (train_step+1), train_loss, r_loss, kl_loss)
    if ((train_step+1) % 5000 == 0):
      vae.save_json("tf_vae/vae.json")

# finished, final model:
vae.save_json("tf_vae/vae.json")
