from logging import raiseExceptions
import gym
import numpy as np

from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env

import torch
from torch import nn
from torch.utils.data import TensorDataset, DataLoader
from torch.utils.tensorboard import SummaryWriter

import matplotlib.pyplot as plt
from pylab import figure, cm

from feedback_env import *
from mlp import MLP

## Helpers

# Create noisy comparison function from reward function
def create_comparison_fn(f, g, sigma_f=0.1, sigma_g=0.5, seed=None):
    np.random.seed(seed)
    def comparison_fn(x, y):
        eps_f, eps_g = np.random.normal(0, sigma_f, size=(1,)), np.random.normal(0, sigma_g, size=(1,))
        r_fn = lambda z: f(z) + eps_f + g(z) + eps_g
        return r_fn(x) > r_fn(y)
    return comparison_fn

def create_reward_fn(f, g, sigma_f=0.1, sigma_g=0.5, seed=None):
    np.random.seed(seed)
    def reward_fn(x):
        eps_f, eps_g = np.random.normal(0, sigma_f, size=(1,)), np.random.normal(0, sigma_g, size=(1,))
        return (f(x) + eps_f + g(x) + eps_g)[0]
    return reward_fn

# Create comparison dataset from sampled actions
def create_comparisons(actions, comparison_fn):
    comparisons_x = []
    comparisons_y = []
    for action1 in actions:
        for action2 in actions:
            x = np.hstack((action1, action2))
            if comparison_fn(action1,action2):
                comparisons_x.append(x.reshape((x.shape[0],)).tolist())
                comparisons_y.append([1,0])
            if comparison_fn(action2, action1):
                comparisons_x.append(x.reshape((x.shape[0],)).tolist())
                comparisons_y.append([0,1])
    
    comparisons_x = np.array(comparisons_x)
    comparisons_y = np.array(comparisons_y)
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
    if policy_cfg["log"]:
        model = PPO("MlpPolicy", env, verbose=verbose, tensorboard_log="./log/policy/"+policy_cfg["log"]+"/")
    else:
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
    print("Learning with Cross Entropy")
    n_sample = reward_cfg['n_sample']
    n_epoch = reward_cfg['n_epoch']
    lr = reward_cfg['lr']
    verbose = reward_cfg['verbose']
    eval_freq = reward_cfg['eval_freq']
    batch_size = reward_cfg['batch_size']
    split = reward_cfg['split']

    if reward_cfg['log']:
        writer = SummaryWriter('./log/reward/'+reward_cfg['log'])

    _, actions = sample(n_sample)
    
    # Generate comparisons dataset
    comparisons_dataset = create_comparisons(actions, comparison_fn)
    train_set, val_set = torch.utils.data.random_split(comparisons_dataset, [int(split*len(comparisons_dataset)), len(comparisons_dataset)-int(split*len(comparisons_dataset))])
    trainloader = torch.utils.data.DataLoader(train_set, batch_size=batch_size, shuffle=True)
    validationloader = torch.utils.data.DataLoader(val_set, batch_size=batch_size, shuffle=True)

    # Create NN
    mlp = MLP()
    ce_loss = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(mlp.parameters(), lr=lr)

    running_loss = 0.0
    # Run the training loop
    for epoch in range(0, n_epoch):
        if verbose: print(f'Starting epoch {epoch+1}')
        for i, data in enumerate(trainloader, 0):
            inputs, targets = data
            output_1 = mlp(inputs[:,0:2])
            output_2 = mlp(inputs[:,2:])
            output = torch.hstack([output_1, output_2])
            # print("inputs", inputs)
            # print("outputs", output)
            # print("targets", targets)
            # print()
            loss = ce_loss(output, targets)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            if i % eval_freq == eval_freq-1:
                running_vloss = 0.0

                mlp.train(False) # Don't need to track gradents for validation
                for j, vdata in enumerate(validationloader, 0):
                    vinputs, vlabels = vdata
                    voutput_1 = mlp(vinputs[:,0:2])
                    voutput_2 = mlp(vinputs[:,2:])
                    voutput = torch.hstack([voutput_1, voutput_2])
                    vloss = ce_loss(voutput, vlabels)
                    running_vloss += vloss.item()
                mlp.train(True) # Turn gradients back on for training

                avg_loss = running_loss / eval_freq
                avg_vloss = running_vloss / len(validationloader)

                # Log the running loss averaged per batch
                if reward_cfg['log']:
                    writer.add_scalars('Training vs. Validation Loss',
                                    { 'Training' : avg_loss, 'Validation' : avg_vloss },
                                    epoch * len(trainloader) + i)
                    writer.flush()
                if verbose:
                    print(f'Epoch {epoch+1} [{i+1}/{len(trainloader)}]\tTrain Loss: {avg_loss:.4f} \tVal Loss: {avg_vloss:.4f}')

                running_loss = 0.0

    # Process is complete.
    if verbose: print('Training process has finished.')

    # Minimize loss

    return lambda x: mlp(torch.tensor(np.array([x])).float()).detach().numpy()[0,0]

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

def visualize_fn(f, title, x_range=[0,1], y_range=[0,1], x_step=0.01, y_step=0.01):
    xs = np.arange(x_range[0], x_range[1], x_step)
    ys = np.arange(y_range[0], y_range[1], y_step)
    z = np.array([[f([float(x),float(y)]) for x in xs] for y in ys])
    plt.imshow(z, extent=[x_range[0],x_range[1],y_range[0], y_range[1]], cmap=cm.jet, origin='lower')
    plt.colorbar()
    plt.title(title)
    plt.show()