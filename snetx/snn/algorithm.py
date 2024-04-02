import math
from typing import Any, Optional, Iterator

import torch
from torch import Tensor
import torch.nn as nn
from torch.nn import Parameter

# surrogate
@torch.jit.script
def spike_emiting(potential_cond):
    """
    """
    return potential_cond.ge(0.0).to(potential_cond)


class basic_surrogate(torch.autograd.Function):
    @staticmethod
    def forward(ctx, inputs, alpha=4.):
        if inputs.requires_grad:
            ctx.save_for_backward(inputs)
            ctx.alpha = alpha
        return spike_emiting(inputs)


class arc_tan(basic_surrogate):
    @staticmethod
    def backward(ctx, grad_out):
        return grad_out * ctx.alpha / 2 / (1 + (math.pi / 2 * ctx.alpha * ctx.saved_tensors[0]).pow_(2)), None


class PiecewiseQuadratic(basic_surrogate):
    @staticmethod
    def backward(ctx, grad_out):
        x, = ctx.saved_tensors
        alpha = ctx.alpha
        x_abs = x.abs()
        mask = (x_abs > (1. / alpha))
        return grad_out * (- (alpha ** 2) * x_abs + alpha).masked_fill_(mask, 0), None


class EvolvedAlpha(object):
    e_max: float = 1.
    epoch: int = 1
    current: int = 0
    def __init__(self, base):
        self.base = base

    def __call__(self):
        return self.base * (1 + EvolvedAlpha.e_max * EvolvedAlpha.current / EvolvedAlpha.epoch)

    @staticmethod
    def step():
        EvolvedAlpha.current += 1
    
    @staticmethod
    def calculate_alpha(alpha):
        return alpha * (1 + EvolvedAlpha.e_max * EvolvedAlpha.current / EvolvedAlpha.epoch)


# 
class Tosnn(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model
        
    def forward(self, x):
        x = x.contiguous()
        timesteps, batch_size = x.shape[:2]
        h = self.model(x.view(-1, *x.shape[2:]))
        return h.view(timesteps, batch_size, *h.shape[1:]).contiguous()
    

# TET Loss
def TETLoss(criterion, target, output, lamb=1e-3):
    r_target = temporal_repeat(target, output.shape[1])
    #
    loss1 = criterion(output.mean(dim=1), target)
    loss2 = criterion(output.flatten(0, 1), r_target.flatten(0, 1))
    return lamb * loss1 + (1 - lamb) * loss2

class TET(object):
    def __init__(self, criterion, lamb=1e-3):
        self.criterion = criterion
        self.lamb = lamb
        
    def __call__(self, y, x):
        return TETLoss(self.criterion, y, x, self.lamb)


# encoding
def temporal_repeat(x, T):
    x = x.unsqueeze(dim=1)
    r_size = [1] * len(x.shape)
    r_size[1] = T
    return x.repeat(*r_size)

class DirectEncoder(object):
    def __init__(self, T):
        self.T = T
    
    def __call__(self, x):
        return temporal_repeat(x, self.T)

