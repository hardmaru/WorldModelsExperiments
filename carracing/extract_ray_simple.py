'''
saves ~ 200 episodes generated from a random policy
'''

import numpy as np
import random
import os
import argparse
import datetime
import cv2
import gym  # @UnresolvedImport
import Box2D  # @UnresolvedImport
import ray  # @UnresolvedImport
import env as myenv

def make_env(self, seed=-1, render_mode=False, full_episode=False):
  return make_env(self.env_name, seed=seed, render_mode=render_mode, full_episode=full_episode)

@ray.remote
def worker(worker_index, max_nrof_trials=200, max_nrof_frames=1000, render_mode=False, dir_name='record'):
    print('%s: starting worker %d' % (gettime(), worker_index))
    
    total_frames = 0
    env = myenv.make_env('carracing', render_mode=render_mode, full_episode=True)
    for _ in range(max_nrof_trials): # 200 trials per worker
      try:
        random_generated_int = random.randint(0, 2**31-1)
        filename = dir_name+"/"+str(random_generated_int)+".npz"
        recording_obs = []
        recording_action = []
    
        np.random.seed(random_generated_int)
        env.seed(random_generated_int)
    
        obs = env.reset() # pixels
    
        for frame in range(max_nrof_frames):
          if render_mode:
            env.render("human")
          else:
            env.render("rgb_array")
            
          obs = cv2.resize(obs, (64, 64), interpolation = cv2.INTER_AREA)  # @UndefinedVariable
          recording_obs.append(obs)
          action = env.action_space.sample()
    
          recording_action.append(action)
          obs, _, done, _ = env.step(action)
    
          if done:
            break
    
        total_frames += (frame+1)
        print("%s: worker %d dead at %d total recorded frames %d" % (gettime(), worker_index, frame+1, total_frames))
        recording_obs = np.array(recording_obs, dtype=np.uint8)
        recording_action = np.array(recording_action, dtype=np.float16)
        np.savez_compressed(filename, obs=recording_obs, action=recording_action)
      except gym.error.Error:
        print("stupid gym error, life goes on")
        env.close()
        env = make_env(render_mode=render_mode)
        continue
    env.close()
    print('%s: worker finished %d' % (gettime(), worker_index))
    
def gettime():
    return datetime.datetime.strftime(datetime.datetime.now(), '%Y%m%d-%H%M%S')
  
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate car racing data from a random policy.")
    parser.add_argument("--nrof_workers", default=4, type=int,
                        help="The number of workers.")
    parser.add_argument("--max_nrof_frames", default=1000, type=str,
                        help="The maximum number of steps to run the environment for.")
    parser.add_argument("--max_nrof_trials", default=200, type=int,
                        help="The maximum number of trials to run.")
    args = parser.parse_args()
    
    for p in [np, gym, Box2D, ray]:
        print('%-12s\t%-10s\t%s' % (p.__name__, p.__version__, p.__file__))
    
    DIR_NAME = os.path.join('record', gettime())
    if not os.path.exists(DIR_NAME):
        os.makedirs(DIR_NAME)
    
    if args.nrof_workers>1:
      ray.init()
      workers = []
      for i in range(args.nrof_workers):
          workers.append(worker.remote(i, render_mode=False, dir_name=DIR_NAME))
      ray.get(workers)
    else:
      worker(0, render_mode=False, dir_name=DIR_NAME)

