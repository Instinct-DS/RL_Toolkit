# Policies to be used for training and evaluation #

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


# MLP POLICIES #
# Q_network #
class Q_network(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_sizes = [256, 256], activation : str = None):
        self.main_head = nn.Sequential()
        activation_map = {
            "relu" : lambda : nn.ReLU(),
            "leakyrelu" : lambda : nn.LeakyReLU(),
            "gelu" : lambda : nn.GELU(),
            "selu" : lambda : nn.SELU(),
            "silu" : lambda : nn.SiLU(),
            "mish" : lambda : nn.Mish(),
            "tanh" : lambda : nn.Tanh(),
            "sigmoid" : lambda : nn.Sigmoid(),
        }
        if activation is not None:
            assert activation in ["relu", "leakyrelu", "gelu", "selu", "silu", "mish", "tanh", "sigmoid"]
        else:
            activation = "relu"

        in_size = state_dim + action_dim
        for i, hidden_size in hidden_sizes:
            self.forward.add_module(name=f"fc{i}", module=nn.Linear(in_features=in_size, out_features=hidden_size))
            self.forward.add_module(name=activation, module=activation_map[activation])
            in_size = hidden_size
        
        self.final = nn.Linear(in_features=in_size, out_features=1)

    def forward(self, state, action):
        x = torch.cat([state, action], dim=-1)
        x = self.main_head(state)
        x = self.final(x)
        return x.squeeze(-1)
    
    
    

