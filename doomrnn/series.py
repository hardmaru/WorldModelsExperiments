'''
Uses pretrained VAE to process dataset to get mu and logvar for each frame, and stores
all the dataset files into one dataset called series/series.npz
'''

import numpy as np
import os
import json
import tensorflow as tf
np.set_printoptions(precision=4, edgeitems=6, linewidth=100, suppress=True)

import random
from doomrnn import reset_graph, ConvVAE

DATA_DIR = "record"
SERIES_DIR = "series"
model_path_name = "tf_vae"

os.environ["CUDA_VISIBLE_DEVICES"]="-1" # disable GPU

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

def encode(img):
  simple_obs = np.copy(img).astype(np.float)/255.0
  simple_obs = simple_obs.reshape(1, 64, 64, 3)
  mu, logvar = vae.encode_mu_logvar(simple_obs)
  z = (mu + np.exp(logvar/2.0) * np.random.randn(*logvar.shape))[0]
  return mu[0], logvar[0], z

def decode(z):
  # decode the latent vector
  img = vae.decode(z.reshape(1, 64)) * 255.
  img = np.round(img).astype(np.uint8)
  img = img.reshape(64, 64, 3)
  return img


# Hyperparameters for ConvVAE
z_size=64
batch_size=1
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
              gpu_mode=False)

vae.load_json(os.path.join(model_path_name, 'vae.json'))

mu_dataset = []
logvar_dataset = []
for i in range(len(dataset)):
  data = dataset[i]
  datalen = len(data)
  mu_data = []
  logvar_data = []
  for j in range(datalen):
    img = data[j]
    mu, logvar, z = encode(img)
    mu_data.append(mu)
    logvar_data.append(logvar)
  mu_data = np.array(mu_data, dtype=np.float16)
  logvar_data = np.array(logvar_data, dtype=np.float16)
  mu_dataset.append(mu_data)
  logvar_dataset.append(logvar_data)
  if (i+1) % 100 == 0:
    print(i+1)

dataset = np.array(dataset)
action_dataset = np.array(action_dataset)
mu_dataset = np.array(mu_dataset)
logvar_dataset = np.array(logvar_dataset)

np.savez_compressed(os.path.join(SERIES_DIR, "series.npz"), action=action_dataset, mu=mu_dataset, logvar=logvar_dataset)
