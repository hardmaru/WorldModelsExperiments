# DoomRNN Experiments

Step by step instructions of reproducing VizDoom TakeCover experiment.

# Citation

If you find this project useful in an academic setting, please cite:

```
@article{Ha2018WorldModels,
  author = {Ha, D. and Schmidhuber, J.},
  title  = {World Models},
  eprint = {arXiv:1803.10122},
  doi    = {10.5281/zenodo.1207631},
  url    = {https://worldmodels.github.io},
  year   = {2018}
}
```

# Notes

-TensorFlow 1.8.0

-NumPy 1.13.3 (1.14 has some annoying warning)

-OpenAI Gym 0.9.4 (breaks for 1.0+ for doom)

-cma 2.2.0, basically 2+ should work

-Python 3, although 2 might work.

-https://github.com/ppaquette/gym-doom (Latest commit 60ff576  on Mar 18, 2017)

-mpi4py 2 (see https://github.com/hardmaru/estool)

# Reading

https://worldmodels.github.io/

http://blog.otoro.net/2017/11/12/evolving-stable-strategies/

http://blog.otoro.net/2017/10/29/visual-evolution-strategies/

# Instructions for running the pretrained model already in repo

To run model in actual environment, and visualize an episode:

`python model.py doomreal norender log/doomrnn.cma.16.64.best.json`

To run model in actual environment 100 times and not visualize the episodes, while computing mean score:

`python model.py doomreal render log/doomrnn.cma.16.64.best.json`

To run model in generated environment, and visualize results:

`python model.py doomrnn norender log/doomrnn.cma.16.64.best.json`

# Instructions for training everything from scratch

Extract 10k random episodes by running the following on a 64-core CPU machine:

`bash extract.bash`

After running this, 12.8k episodes should be saved as npz files in `record`. We will only use 10k episodes.

On a machine with a GPU, run the following to train VAE for 10 epochs:

`python vae_train.py`

After training, the vae model saved in `tf_vae/vae.json`

Next, we pre-process collected data using pre-trained VAE by running:

`python series.py`

A new dataset will be created in `series`. After this is recorded, train the MDN-RNN by running:

`train rnn using python rnn_train.py`

This will produce a model in `tf_rnn/rnn.json` and also `initial_z.json`.

You must now copy copy vae.json, initial_z.json and rnn.json over to `tf_models` directory and overwrite previous files if they were there.

Now on a 64-core CPU machine, run the CMA-ES based training:

`python train.py`

You can monitor progress using the `plot_training_progress.ipynb` notebook which loads the `log` files being generated. After 200 generations (or around 4-5 hours), it should be enough, and you can test the model by running:

`python model.py doomreal norender log/doomrnn.cma.16.64.best.json`

# Licence

MIT
