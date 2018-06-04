'''
Uses pretrained VAE to process dataset to get mu and logvar for each frame, and stores
all the dataset files into one dataset called series/series.npz
'''

import numpy as np
import os
import json
import tensorflow as tf
import random
from vae.vae import ConvVAE, reset_graph

os.environ["CUDA_VISIBLE_DEVICES"]="0"

DATA_DIR = "record"
SERIES_DIR = "series"
model_path_name = "tf_vae"

if not os.path.exists(SERIES_DIR):
    os.makedirs(SERIES_DIR)

def load_raw_data_list(filelist):
  data_list = []
  action_list = []
  counter = 0
  for i in range(len(filelist)):
    filename = filelist[i]
    raw_data = np.load(os.path.join(DATA_DIR, filename))
    data_list.append(raw_data['obs'])
    action_list.append(raw_data['action'])
    if ((i+1) % 1000 == 0):
      print("loading file", (i+1))
  return data_list, action_list

def encode_batch(batch_img):
  simple_obs = np.copy(batch_img).astype(np.float)/255.0
  simple_obs = simple_obs.reshape(batch_size, 64, 64, 3)
  mu, logvar = vae.encode_mu_logvar(simple_obs)
  z = (mu + np.exp(logvar/2.0) * np.random.randn(*logvar.shape))
  return mu, logvar, z

def decode_batch(batch_z):
  # decode the latent vector
  batch_img = vae.decode(z.reshape(batch_size, z_size)) * 255.
  batch_img = np.round(batch_img).astype(np.uint8)
  batch_img = batch_img.reshape(batch_size, 64, 64, 3)
  return batch_img


# Hyperparameters for ConvVAE
z_size=32
batch_size=1000 # treat every episode as a batch of 1000!
learning_rate=0.0001
kl_tolerance=0.5

filelist = os.listdir(DATA_DIR)
filelist.sort()
filelist = filelist[0:10000]

dataset, action_dataset = load_raw_data_list(filelist)

reset_graph()

vae = ConvVAE(z_size=z_size,
              batch_size=batch_size,
              learning_rate=learning_rate,
              kl_tolerance=kl_tolerance,
              is_training=False,
              reuse=False,
              gpu_mode=True) # use GPU on batchsize of 1000 -> much faster

vae.load_json(os.path.join(model_path_name, 'vae.json'))

mu_dataset = []
logvar_dataset = []
for i in range(len(dataset)):
  data_batch = dataset[i]
  mu, logvar, z = encode_batch(data_batch)
  mu_dataset.append(mu.astype(np.float16))
  logvar_dataset.append(logvar.astype(np.float16))
  if ((i+1) % 100 == 0):
    print(i+1)

action_dataset = np.array(action_dataset)
mu_dataset = np.array(mu_dataset)
logvar_dataset = np.array(logvar_dataset)

np.savez_compressed(os.path.join(SERIES_DIR, "series.npz"), action=action_dataset, mu=mu_dataset, logvar=logvar_dataset)
