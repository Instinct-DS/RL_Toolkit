# Utils files to have the helper functions #

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

# Find atanh #
def atanh(x, eps=1e-6):
    x = x.clamp(-1 + eps, 1 - eps)
    return 0.5 * (torch.log1p(x) - torch.log1p(-x))

# Soft Update #
def SoftUpdate(target_network, source_network, tau=0.005):
    for target_param, source_param in zip(target_network.parameters(), source_network.parameters()):
        target_param.data.copy_(
            (1 - tau) * target_param.data + tau * source_param.data
        )

