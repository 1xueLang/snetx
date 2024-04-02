from typing import Iterator
import torch
import torch.nn as nn
from torch import Tensor
from snetx.snn.algorithm import Tosnn

__all__ = ['tdBN2d', 'TEBN2d', 'SynctbBN', 'SyncTEBN']

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
        self.bn = Tosnn(nn.BatchNorm2d(num_features, eps, momentum, affine, track_running_stats))
        self.affine = affine
        self.initialized = (affine == False)
        self.weight = None
        
    def forward(self, input: Tensor) -> Tensor:
        if not self.initialized:
            self.bn_init(input)
            return input
        normx = self.bn(input)
        if self.affine:
            return normx * self.weight
        else:
            return normx
    
    def bn_init(self, x):
        self.initialized = True
        if self.affine:
            c_size = [1] * len(x.shape)
            c_size[1] = x.shape[1]
            self.weight = nn.Parameter(torch.ones(c_size)).to(x)
    
    def parameters(self, recurse: bool = True) -> Iterator[nn.Parameter]:
        if not self.initialized:
            raise NotImplementedError('TEBN is not initialized.')
        return super().parameters(recurse)

 
class SyncTEBN(nn.Module):
    def __init__(self, num_features, eps=0.00001, momentum=0.1, affine=True, track_running_stats=True):
        super().__init__()
        self.bn = Tosnn(nn.SyncBatchNorm(num_features, eps, momentum, affine, track_running_stats))
        self.affine = affine
        self.initialized = (affine == False)
        self.weight = None
        
    def forward(self, input: Tensor) -> Tensor:
        if not self.initialized:
            self.bn_init(input)
            return input
        normx = self.bn(input)
        if self.affine:
            return normx * self.weight
        else:
            return normx
    
    def bn_init(self, x):
        self.initialized = True
        if self.affine:
            c_size = [1] * len(x.shape)
            c_size[1] = x.shape[1]
            self.weight = nn.Parameter(torch.ones(c_size)).to(x)
    
    def parameters(self, recurse: bool = True) -> Iterator[nn.Parameter]:
        if not self.initialized:
            raise NotImplementedError('TEBN is not initialized.')
        return super().parameters(recurse)
