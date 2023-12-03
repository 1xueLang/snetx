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
    return potential_cond.gt(0.0).to(potential_cond)


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

    
class tdBN2d(nn.BatchNorm2d):
    def __init__(self, num_features, eps=0.00001, momentum=0.1, affine=True, track_running_stats=True):
        super().__init__(num_features, eps, momentum, affine, track_running_stats)
        
    def forward(self, x: Tensor) -> Tensor:
        timesteps, batch_size = x.shape[:2]
        h = super().forward(x.view(-1, *x.shape[2:]))
        return h.view(timesteps, batch_size, *h.shape[1:])

class SynctdBN(nn.SyncBatchNorm):
    def __init__(self, num_features: int, eps: float = 0.00001, momentum: float = 0.1, affine: bool = True, track_running_stats: bool = True, process_group=None) -> None:
        super().__init__(num_features, eps, momentum, affine, track_running_stats, process_group)
        
    def forward(self, x: Tensor) -> Tensor:
        timesteps, batch_size = x.shape[:2]
        h = super().forward(x.view(-1, *x.shape[2:]))
        return h.view(timesteps, batch_size, *h.shape[1:])
    
class TEBN2d(nn.Module):
    def __init__(self, num_features, eps=0.00001, momentum=0.1, affine=True, track_running_stats=True):
        super().__init__()
        self.bn = Tosnn(nn.BatchNorm2d(num_features, eps, momentum, False, track_running_stats))
        self.affine = affine
        self.initialized = (affine == False)
        self.weight = None
        self.bias = None
        
    def forward(self, input: Tensor) -> Tensor:
        if not self.initialized:
            self.bn_init(input)
            return input
        normx = self.bn(input)
        if self.affine:
            return normx * self.weight + self.bias
        else:
            return normx
    
    def bn_init(self, x):
        self.initialized = True
        if self.affine:
            c_size = [1] * len(x.shape)
            c_size[1], c_size[2] = x.shape[1], x.shape[2]
            self.weight = Parameter(torch.ones(c_size)).to(x)
            self.bias = Parameter(torch.zeros(c_size)).to(x)
    
    def parameters(self, recurse: bool = True) -> Iterator[Parameter]:
        if not self.initialized:
            raise NotImplementedError('TEBN is not initialized.')
        return super().parameters(recurse)

 
class SyncTEBN(nn.Module):
    def __init__(self, num_features, eps=0.00001, momentum=0.1, affine=True, track_running_stats=True):
        super().__init__()
        self.bn = Tosnn(nn.SyncBatchNorm(num_features, eps, momentum, False, track_running_stats))
        self.affine = affine
        self.initialized = (affine == False)
        self.weight = None
        self.bias = None
        
    def forward(self, input: Tensor) -> Tensor:
        if not self.initialized:
            self.bn_init(input)
            return input
        normx = self.bn(input)
        if self.affine:
            return normx * self.weight + self.bias
        else:
            return normx
    
    def bn_init(self, x):
        self.initialized = True
        if self.affine:
            c_size = [1] * len(x.shape)
            c_size[1], c_size[2] = x.shape[1], x.shape[2]
            self.weight = Parameter(torch.ones(c_size)).to(x)
            self.bias = Parameter(torch.zeros(c_size)).to(x)
    
    def parameters(self, recurse: bool = True) -> Iterator[Parameter]:
        if not self.initialized:
            raise NotImplementedError('TEBN is not initialized.')
        return super().parameters(recurse)