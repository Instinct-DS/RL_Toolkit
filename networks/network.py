# Policies to be used for training and evaluation #

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


# MLP POLICIES #
# Q_network #
class Q_network(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_sizes = [256, 256], activation : str = None, layer_norm=False):
        super().__init__()
        self.main_head = nn.Sequential()
        activation_map = {
            "relu" : nn.ReLU(),
            "leakyrelu" : nn.LeakyReLU(),
            "gelu" : nn.GELU(),
            "selu" : nn.SELU(),
            "silu" : nn.SiLU(),
            "mish" : nn.Mish(),
            "tanh" : nn.Tanh(),
            "sigmoid" : nn.Sigmoid(),
        }
        if activation is not None:
            assert activation in ["relu", "leakyrelu", "gelu", "selu", "silu", "mish", "tanh", "sigmoid"]
        else:
            activation = "relu"

        in_size = state_dim + action_dim
        for i, hidden_size in enumerate(hidden_sizes):
            self.main_head.add_module(name=f"{2*i}", module=nn.Linear(in_features=in_size, out_features=hidden_size))
            if layer_norm:
                self.main_head.add_module(name="layer_norm", module=nn.LayerNorm(in_size))
            self.main_head.add_module(name=f"{2*i+1}", module=activation_map[activation])
            in_size = hidden_size
        
        self.main_head.add_module(name=f"{2*len(hidden_sizes)}", module=nn.Linear(in_features=in_size, out_features=1))

    def forward(self, state, action):
        x = torch.cat([state, action], dim=-1)
        x = self.main_head(x)
        return x.squeeze(-1)
    
    
    

