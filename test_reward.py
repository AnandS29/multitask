import gym
import numpy as np

from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env

import torch
from torch import nn
from torch.utils.data import TensorDataset, DataLoader

import matplotlib.pyplot as plt

from mlp import *
from feedback_env import *
from learning import *

reward_cfg = {
    'n_sample': 100,
    'n_epoch': 20,
    'lr': 0.5,
    'verbose': False,
    'eval_freq': 100,
    'batch_size': 128,
    'split': 0.8,
    'log': 'cross_ent'
}

# Create reward function
f = lambda x: -(x[0]-0.5)**2
g = lambda x: -(x[1]-0.5)**2
sigma_f, sigma_g = 0.0, 0.0
reward_fn = create_reward_fn(f, g, sigma_f, sigma_g)
comparison_fn = create_comparison_fn(f, g, sigma_f, sigma_g)

# Initialize random sampler
sample = lambda n: (None, [np.random.uniform(0, 1, size=(2,)) for _ in range(n)])

# Learn reward function from samples and feedback from comparison function
learnt_reward_fn = learn_reward(sample, comparison_fn, reward_cfg=reward_cfg)
    
# Visualize reward functions (true and learnt)
visualize_fn(reward_fn, "True reward function")
visualize_fn(learnt_reward_fn, "Learnt reward function")