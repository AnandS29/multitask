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

# Setup configs for reward and policy learning
def make_reward_cfg(n):
    reward_cfg = {
        'n_sample': n,
        'n_epoch': 40,
        'lr': 0.001,
        'verbose': True,
        'eval_freq': 100,
        'batch_size': 64,
        'split': 0.8,
        'log': False,
        'action_dim': 1,
        'layers': [5,5,5]
    }
    return reward_cfg


## Setup

n_samples = [50, 100, 150, 200, 250, 300, 350, 400]

# Create comparison function
f = lambda x: x[0]
var_1, var_2 = 0.0, 10.0
reward_fn_true = lambda x : f(x)
seed = None
np.random.seed(seed)
noise_fn = lambda x: step_noise(x[0], x_step=0.8, var_1=var_1, var_2=var_2, seed=seed)
reward_fn_true_noisy = create_reward_fn_1(f, noise_fn, seed=seed)
comparison_fn = create_comparison_fn_1(f, noise_fn, seed=seed)

# Initialize random sampler
sample = lambda n: (None, [np.random.uniform(0, 1, size=(1,)) for _ in range(n)])

# Plot true reward function
vis_fn = lambda fn, title: visualize_fn_1(fn, title=title, x_range=[0,1], x_step=0.01)
vis_fn(reward_fn_true, title="True reward function")
vis_fn(reward_fn_true_noisy, title="True (noisy) reward function")

## Run feedback loop

comparisons_data = None
samples = {}
reward_fns = {}

for n in n_samples:
    print(f"Number of samples = {n}", end=",")
    # Learn reward function from samples and feedback from comparison function
    reward_cfg = make_reward_cfg(n)
    reward_fn, comparisons_data = learn_reward(sample, comparison_fn, reward_cfg=reward_cfg, prev_comparisons=comparisons_data)

    # Store reward function
    reward_fns[n] = reward_fn

    # plot_sampler_1(samples[i], title=str(i), n=1000)
    vis_fn(reward_fns[n], title=str(n))

for n in n_samples:
    plot_fn_1(reward_fns[n], title=str(n), save=True)