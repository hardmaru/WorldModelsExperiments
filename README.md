# World Models Experiments

Step by step instructions of reproducing World Models

# Notes

TensorFlow 1.8.0
NumPy 1.13.3 (1.14 has some annoying warning)
OpenAI Gym 0.9.4 (breaks for 1.0+ for doom)
cma 2.2.0
Python 3
https://github.com/ppaquette/gym-doom (Latest commit 60ff576  on Mar 18, 2017)
mpi4py 2 (see https://github.com/hardmaru/estool)

# Reading

https://worldmodels.github.io/
http://blog.otoro.net/2017/11/12/evolving-stable-strategies/
http://blog.otoro.net/2017/10/29/visual-evolution-strategies/

# DoomRNN

bash extract.bash # doom doesn't use OpenGL, so no need for headless X

Train VAE for 10 epochs:
python vae_train.py

vae model saved in tf_vae/vae.json

Pre-process collected data using pre-trained VAE by running:
python series.py

train rnn using python rnn_train.py

copy this vae.json over to tf_models/vae.json
copy initial_z.json and rnn.json over to prod

`python model.py doomreal norender log/doomrnn.cma.16.64.best.json 0`

# Car Racing

To generate 12800 random episodes, record them in record/[key].npz. Note we will only use 10k episodes to train VAE and MDN-RNN.

Point out that VM machines with GPUs have problem using OpenGL levels.

For mac, point out setting resolution higher.

bash extract.bash # uses headless X server. might not be needed.

Train VAE for 10 epochs:
python vae_train.py

vae model saved in tf_vae/vae.json

run python series.py to preprocess this dataset

train rnn using python rnn_train.py

copy vae.json over to vae/vae.json

copy initial_z.json and rnn.json over to prod

bash gce_train.bash #python train.py # note num_worker * num_worker_trial should be 64 # run in X

# final verify

python model.py norender log/carracing.cma.16.64.best.json 0

