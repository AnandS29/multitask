import gym
import numpy as np

from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env

import torch
from torch import nn
from torch.utils.data import TensorDataset, DataLoader

import matplotlib.pyplot as plt

class MLP(nn.Module):
  '''
    Multilayer Perceptron.
  '''
  def __init__(self):
    super().__init__()
    self.layers = nn.Sequential(
      nn.Linear(2, 5),
      nn.ReLU(),
      nn.Linear(5, 5),
      nn.ReLU(),
      nn.Linear(5, 5),
      nn.ReLU(),
      nn.Linear(5, 1)
    )


  def forward(self, x):
    '''Forward pass'''
    return self.layers(x)