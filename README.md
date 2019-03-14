# World Models Experiments

Step by step instructions of reproducing [World Models](https://worldmodels.github.io/) ([pdf](https://arxiv.org/abs/1803.10122)).

![World Models](https://worldmodels.github.io/assets/world_models_card_both.png)

Please see [blog post](http://blog.otoro.net//2018/06/09/world-models-experiments/) for step-by-step instructions.

# Note regarding OpenAI Gym Version

Please note the library versions in the blog post. In particular, the experiments work on gym 0.9.x and does NOT work on gym 0.10.x. You can install the older version of gym using the command `pip install gym==0.9.4`, `pip install numpy==1.13.3` etc.

# Citation

If you find this project useful in an academic setting, please cite:

```latex
@incollection{ha2018worldmodels,
  title = {Recurrent World Models Facilitate Policy Evolution},
  author = {Ha, David and Schmidhuber, J{\"u}rgen},
  booktitle = {Advances in Neural Information Processing Systems 31},
  pages = {2451--2463},
  year = {2018},
  publisher = {Curran Associates, Inc.},
  url = {https://papers.nips.cc/paper/7512-recurrent-world-models-facilitate-policy-evolution},
  note = "\url{https://worldmodels.github.io}",
}
```

# Issues

For general discussion about the World Model article, there are already some good discussion threads here in the GitHub [issues](https://github.com/worldmodels/worldmodels.github.io/issues) page of the interactive article. Please raise issues about this specific implementation in the [issues](https://github.com/hardmaru/WorldModelsExperiments/issues) page of this repo.

# Licence

MIT
