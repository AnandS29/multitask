import gym
import numpy as np

from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env

import torch
from torch import nn
from torch.utils.data import TensorDataset, DataLoader

import matplotlib.pyplot as plt

class FeedbackEnv(gym.Env):
    def __init__(self, reward_fn, action_dim):
        super(FeedbackEnv, self).__init__()
        self.action_space = gym.spaces.Box(low=-10, high=10, shape=(action_dim,))
        self.observation_space = gym.spaces.Box(low=0, high=0, shape=(1,), dtype=np.uint8) # No obs
        self.reward_fn = reward_fn

    def step(self, action):
        reward = self.reward_fn(action)
        observation = np.array([0])
        done = True
        info = {}
        return observation, reward, done, info

    def reset(self):
        observation = np.array([0])
        return observation
        
    def render(self):
        raise NotImplementedError
    # def close (self):
    #     ...

def register_fb_env(r_fn, action_dim):
    gym.envs.register(
        id='FeedbackEnv-v0',
        entry_point='__main__:FeedbackEnv',
        max_episode_steps=150,
        kwargs={
            'reward_fn' : r_fn,
            'action_dim' : action_dim
        }
    )