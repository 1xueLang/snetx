import torch
import torch.nn as nn

import snetx.snn.algorithm as snnalgo

class LIF(nn.Module):
    r""" STBP based Leaky Integrate and Fire neuron model.
    
    Args:
        tau (float): Exponential attenuation coefficient of the membrane potential in LIF model. Default: 2.0.
    """
    def __init__(
        self, 
        tau: float = 2.0, 
        v_th: float = 1.0, 
        v_reset: float = 0.0, 
        sg: torch.autograd.Function = snnalgo.PiecewiseQuadratic, 
        alpha = lambda : 1.0,
        detach: bool = False
    ) -> None:
        super().__init__()
        self.tau = tau
        self.v_th = v_th
        self.v_reset = v_reset
        self.sg = sg
        self.alpha = alpha
        self.detach = detach
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        u = 0
        out = torch.zeros_like(x)
        for i in range(x.shape[1]):
            u = u / self.tau + x[:, i]
            s = self.sg.apply(u - self.v_th, self.alpha())
            out[:, i] = s
            if self.detach:
                s = s.detach()
            u = u * (1 - s) + self.v_reset * s
            
        return out