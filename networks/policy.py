# Policies to be used for training and evaluation #

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import numpy as np

# HELPER FUNCTIONS #
def atanh(x, eps=1e-6):
    x = x.clamp(-1 + eps, 1 - eps)
    return 0.5 * (torch.log1p(x) - torch.log1p(-x))

# MLP POLICIES #
# TanhGaussianPolicy #
class TanhGaussianPolicy(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_sizes = [256, 256], activation : str = None):
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

        in_size = state_dim
        for i, hidden_size in enumerate(hidden_sizes):
            self.main_head.add_module(name=f"fc{i}", module=nn.Linear(in_features=in_size, out_features=hidden_size))
            self.main_head.add_module(name=activation, module=activation_map[activation])
            in_size = hidden_size
        
        self.mu_head = nn.Linear(in_features=in_size, out_features=action_dim)
        self.log_std_head = nn.Linear(in_features=in_size, out_features=action_dim)

    def forward(self, state):
        x = self.main_head(state)
        mu = self.mu_head(x)
        log_std = self.log_std_head(x)
        log_std = torch.clamp(log_std, -5, 2)
        return mu, log_std
    
    def sample(self, state):
        mu, log_std = self.forward(state)
        std = torch.exp(log_std)

        dist = torch.distributions.Normal(mu, std)
        action_pre = dist.sample()
        action = torch.tanh(action_pre)
        log_prob_ = dist.log_prob(action_pre).sum(dim=-1)
        log_det_jac = torch.sum(torch.log(1 - action**2 + 1e-8), dim=-1)
        log_prob = log_prob_ - log_det_jac
        return action, log_prob
    
    def sample_det(self, state):
        mu, log_std = self.forward(state)
        std = torch.exp(log_std)

        dist = torch.distributions.Normal(mu, std)
        action_pre = mu
        action = torch.tanh(action_pre)
        log_prob_ = dist.log_prob(action_pre).sum(dim=-1)
        log_det_jac = torch.sum(torch.log(1 - action**2 + 1e-8), dim=-1)
        log_prob = log_prob_ - log_det_jac
        return action, log_prob
    
    def log_probs(self, state, action):
        action_pre = atanh(action)
        mu, log_std = self.forward(state)
        std = torch.exp(log_std)

        dist = torch.distributions.Normal(mu, std)
        log_prob_ = dist.log_prob(action_pre).sum(dim=-1)
        log_det_jac = torch.sum(2 * (math.log(2) - action_pre - F.softplus(-2 * action_pre)), dim=-1)
        log_prob = log_prob_ - log_det_jac
        return log_prob

