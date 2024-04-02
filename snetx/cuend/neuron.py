import torch
import torch.nn as nn

from .nnfunc import stbpLIF, detachedLIF

REGISTERED_SG_FUNCLIST = {
    'LeakyReLU': 1,
    'sigmoid': 2,
    'arctan': 3,
    'PiecewiseQuadratic': 4
}

# 
class BasicCuNeuron(nn.Module):
    r""" Basic cupy neuron class with some essential parameters.
    
    Args:
        v_th (float): Threshold of membrane potential to emit spikes. Default: 1.0.
        v_reset (float): Reset membrane potential after spiking. Default: 0.0.
        sg (str): Surrogate function index, see REGISTERED_SG_FUNCLIST. Default: 'arctan'.
        alpha (float): Parameter of surrogate function, which controls the function shape. Default: 4.0.
    """
    def __init__(
        self, 
        v_th: float = 1.0, 
        v_reset: float = 0.0, 
        sg: str = 'PiecewiseQuadratic', 
        alpha: float = 1.0
    ) -> None:
        super().__init__()
        self.v_th = v_th
        self.v_reset = v_reset
        self.suro = REGISTERED_SG_FUNCLIST[sg]
        self.alpha = alpha
        self.cu_neuron_func = None

class LIF(BasicCuNeuron):
    r""" STBP based Leaky Integrate and Fire neuron model.
    
    Args:
        tau (float): Exponential attenuation coefficient of the membrane potential in LIF model. Default: 2.0.
    """
    def __init__(
        self, 
        tau: float = 2.0, 
        v_th: float = 1.0, 
        v_reset: float = 0.0, 
        sg: str = 'PiecewiseQuadratic', 
        alpha = lambda : 1.0,
        detach: bool = False
    ) -> None:
        super().__init__(v_th, v_reset, sg, alpha)
        self.tau = torch.tensor(tau)
        self.cu_neuron_func = stbpLIF if not detach else detachedLIF
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.cu_neuron_func.apply(
            x, self.tau, self.v_th, self.v_reset, self.suro, self.alpha()
        )
    
class IF(LIF):
    def __init__(
        self, 
        v_th: float = 1.0, 
        v_reset: float = 0.0, 
        sg: str = 'PiecewiseQuadratic', 
        alpha = lambda : 1.0, 
        detach: bool = False
    ) -> None:
        super().__init__(1.0, v_th, v_reset, sg, alpha, detach)