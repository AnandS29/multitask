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
  def __init__(self, input_dim, layer_dims):
    super().__init__()
    layers = [nn.Linear(input_dim, layer_dims[0])]
    for i in range(len(layer_dims)-1):
      layers.append(nn.ReLU())
      layers.append(nn.Linear(layer_dims[i], layer_dims[i+1]))
    layers.append(nn.ReLU())
    layers.append(nn.Linear(layer_dims[-1], 1))

    self.layers = nn.Sequential(*layers)


  def forward(self, x):
    '''Forward pass'''
    return self.layers(x)