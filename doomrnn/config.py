from collections import namedtuple

Game = namedtuple('Game', ['env_name', 'input_size', 'output_size', 'activation'])

games = {}

doomrnn = Game(env_name='doomrnn',
  input_size=576+512*1,
  output_size=1,
  activation='tanh',
)
games['doomrnn'] = doomrnn

doomreal = Game(env_name='doomreal',
  input_size=576+512*1,
  output_size=1,
  activation='tanh',
)
games['doomreal'] = doomreal
