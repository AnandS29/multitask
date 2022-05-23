import gym
import numpy as np

from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env

import torch
from torch import nn
from torch.utils.data import TensorDataset, DataLoader

import matplotlib.pyplot as plt

from feedback_env import *
from mlp import MLP

## Helpers

# Create noisy comparison function from reward function
def create_comparison_fn(f, g, sigma_f=0.1, sigma_g=0.5, seed=None):
    np.random.seed(seed)
    def comparison_fn(x, y):
        eps_f, eps_g = np.random.normal(0, sigma_f, size=(1,)), np.random.normal(0, sigma_g, size=(1,))
        r_fn = lambda x: f(x) + eps_f + g(x) + eps_g
        return r_fn(x) >= r_fn(y)
    return comparison_fn

# Create comparison dataset from sampled actions
def create_comparisons(actions, comparison_fn):
    comparisons_x = []
    comparisons_y = []
    for action1 in actions:
        for action2 in actions:
            if comparison_fn(action1,action2):
                comparisons_x.append(action1)
                comparisons_y.append(np.array([1]))
                comparisons_x.append(action2)
                comparisons_y.append(np.array([0]))
            else:
                comparisons_x.append(action2)
                comparisons_y.append(np.array([1]))
                comparisons_x.append(action1)
                comparisons_y.append(np.array([0]))
                
    tensor_x = torch.Tensor(comparisons_x)
    tensor_y = torch.Tensor(comparisons_y)
    comparisons = TensorDataset(tensor_x,tensor_y)
    
    return comparisons

## Learning code

# Train policy to optimize given reward function, and returns a sampler from 
# which one can sample actions and rewards from the policy
def train_policy(r_fn, policy_cfg):
    # Create env with given reward function
    register_fb_env(r_fn)

    if policy_cfg["verbose"]:
        verbose = 1
    else:
        verbose = 0

    timesteps = policy_cfg["timesteps"]
    algo = policy_cfg["algo"]

    print("Learning with PPO")
    env = make_vec_env("FeedbackEnv-v0", n_envs=1)
    model = PPO("MlpPolicy", env, verbose=verbose)
    model.learn(total_timesteps=timesteps)
    # model.save("ppo_fb")

    # Create sampler
    def sample(n):
        obs = env.reset()
        i = 0
        rewardss = []
        actions = []
        while i < n:
            action, _states = model.predict(obs)
            obs, rewards, dones, info = env.step(action)
            rewardss.append(rewards)
            actions.append(action)
            i += 1
            # env.render()
        rewardss = np.ndarray.flatten(np.array(rewardss))
        return rewardss, [a.reshape(2,) for a in actions]

    return sample

# Learns reward function minimize the cross entropy loss on comparison dataset
def learn_reward(sample, comparison_fn, reward_cfg):
    n_sample = reward_cfg['n_sample']
    n_epoch = reward_cfg['n_epoch']
    lr = reward_cfg['lr']
    verbose = reward_cfg['verbose']
    print_freq = reward_cfg['print_freq']
    batch_size = reward_cfg['batch_size']

    _, actions = sample(n_sample)
    
    # Generate comparisons dataset
    comparisons_dataset = create_comparisons(actions, comparison_fn)
    trainloader = torch.utils.data.DataLoader(comparisons_dataset, batch_size=1, shuffle=True)

    # Create NN
    mlp = MLP()
    loss_function = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(mlp.parameters(), lr=lr)

    # Run the training loop
    for epoch in range(0, n_epoch): # 5 epochs at maximum
        if verbose: print(f'Starting epoch {epoch+1}')
        current_loss = 0.0
        for i, data in enumerate(trainloader, 0):
            inputs, targets = data
            optimizer.zero_grad()
            outputs = mlp(inputs)
            loss = loss_function(outputs, targets)
            loss.backward()
            optimizer.step()
            
            # Print statistics
            current_loss += loss.item()
            if verbose and i % print_freq == print_freq-1:
                print('Loss after mini-batch %5d: %.3f' %
                    (i + 1, current_loss / print_freq), end= ", ")
                current_loss = 0.0

    # Process is complete.
    if verbose: print('Training process has finished.')

    # Minimize loss

    return lambda x: mlp(torch.tensor(x)).detach().numpy()[0]

## Visualization

# Visualize samples from policy
def plot_sampler(sample, title, n=100):
    _, actions = sample(n)
    x = [a[0] for a in actions]
    y = [a[1] for a in actions]
    plt.figure(figsize=(2,2))
    plt.title(title)
    plt.scatter(x, y)
    plt.show()
    print(np.average(actions,axis=0),np.std(actions,axis=0))