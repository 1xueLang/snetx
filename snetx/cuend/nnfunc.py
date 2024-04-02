import os
from typing import Union, Optional, List

import cupy as cp
import torch

from torch.utils.dlpack import to_dlpack as tens2dlpack


_CURPATH = os.path.abspath(__file__)[:-9]

with open(os.path.join(_CURPATH, 'C/neuron.cu'), 'r') as f:
    CU_SOURCE_CODE_RAW_STRING = f.read()


def tensor_to_cparray(ten: torch.Tensor) -> cp.ndarray:
    if hasattr(cp, 'core'):
        return cp.core.dlpack.fromDlpack(tens2dlpack(ten))
    else:
        return cp.from_dlpack(tens2dlpack(ten))


class stbpLIF(torch.autograd.Function):
    
    funclists = ['Leaky_integrate_FP<float>', 'Leaky_integrate_BP<float>']
    
    cu_module = cp.RawModule(code=CU_SOURCE_CODE_RAW_STRING,
                             options=('-std=c++11', '-I ' + _CURPATH),
                             name_expressions=funclists)
    
    neuron_FP = cu_module.get_function(funclists[0])
    neuron_BP = cu_module.get_function(funclists[1])
    
    @staticmethod
    def forward(
        ctx, 
        inputs: torch.Tensor, 
        tau: float, 
        v_th: float, 
        v_reset: float, 
        suro: int, 
        alpha: float
    ) -> torch.Tensor:
        r"""
        Args:
            inputs (Tensor): Input from pre-synapses.
            tau: LIF neuron model's parameter \tau.
            v_th: Firing threshold.
            v_reset: Reset potential after firing.
            suro: Number of surrogate function.
            alpha: Parameter of surrogate function.
        Returns:
            spikes (Tensor): Output spike train. 
        """
        psps = torch.zeros_like(inputs)
        spikes = torch.zeros_like(inputs)
        ctx.batch, ctx.T = inputs.shape[:2]
        ctx.dim = inputs[0][0].numel()
        ctx.tau = tau
        ctx.v_th = v_th
        ctx.suro = suro
        ctx.alpha = alpha
        with cp.cuda.Device(psps.get_device()):
            stbpLIF.neuron_FP(((1023 + ctx.batch * ctx.dim) // 1024,), (1024,), (
                tensor_to_cparray(inputs.contiguous()),
                tensor_to_cparray(psps),
                tensor_to_cparray(spikes),
                cp.float32(tau),
                cp.float32(v_th),
                cp.float32(v_reset),
                cp.int32(ctx.batch), 
                cp.int32(ctx.T), 
                cp.int32(ctx.dim))
            )
        ctx.save_for_backward(psps)
        return spikes
    
    @staticmethod
    def backward(ctx, grad_out: torch.Tensor) -> List[Optional[torch.Tensor]]:
        psps, = ctx.saved_tensors
        grad_x = torch.zeros_like(psps)
        with cp.cuda.Device(psps.get_device()):
            stbpLIF.neuron_BP(((1023 + ctx.batch * ctx.dim) // 1024,), (1024,), (
                tensor_to_cparray(psps),
                tensor_to_cparray(grad_out.contiguous()),
                tensor_to_cparray(grad_x),
                cp.float32(ctx.tau),
                cp.float32(ctx.v_th),
                cp.int32(ctx.batch), 
                cp.int32(ctx.T), 
                cp.int32(ctx.dim),
                cp.int32(ctx.suro), 
                cp.float32(ctx.alpha))
            )
        return grad_x, None, None, None, None, None
    
class detachedLIF(torch.autograd.Function):
    
    funclists = ['Leaky_integrate_FP<float>', 'Leaky_integrate_detached_BP<float>']
    
    cu_module = cp.RawModule(code=CU_SOURCE_CODE_RAW_STRING,
                             options=('-std=c++11', '-I ' + _CURPATH),
                             name_expressions=funclists)
    
    neuron_FP = cu_module.get_function(funclists[0])
    neuron_BP = cu_module.get_function(funclists[1])
    
    @staticmethod
    def forward(
        ctx, 
        inputs: torch.Tensor, 
        tau: float, 
        v_th: float, 
        v_reset: float, 
        suro: int, 
        alpha: float
    ) -> torch.Tensor:
        r"""
        Args:
            inputs (Tensor): Input from pre-synapses.
            tau: LIF neuron model's parameter \tau.
            v_th: Firing threshold.
            v_reset: Reset potential after firing.
            suro: Number of surrogate function.
            alpha: Parameter of surrogate function.
        Returns:
            spikes (Tensor): Output spike train. 
        """
        psps = torch.zeros_like(inputs)
        spikes = torch.zeros_like(inputs)
        ctx.batch, ctx.T = inputs.shape[:2]
        ctx.dim = inputs[0][0].numel()
        ctx.tau = tau
        ctx.v_th = v_th
        ctx.suro = suro
        ctx.alpha = alpha
        with cp.cuda.Device(psps.get_device()):
            detachedLIF.neuron_FP(((1023 + ctx.batch * ctx.dim) // 1024,), (1024,), (
                tensor_to_cparray(inputs.contiguous()),
                tensor_to_cparray(psps),
                tensor_to_cparray(spikes),
                cp.float32(tau),
                cp.float32(v_th),
                cp.float32(v_reset),
                cp.int32(ctx.batch), 
                cp.int32(ctx.T), 
                cp.int32(ctx.dim))
            )
        ctx.save_for_backward(psps)
        return spikes
    
    @staticmethod
    def backward(ctx, grad_out: torch.Tensor) -> List[Optional[torch.Tensor]]:
        psps, = ctx.saved_tensors
        grad_x = torch.zeros_like(psps)
        with cp.cuda.Device(psps.get_device()):
            detachedLIF.neuron_BP(((1023 + ctx.batch * ctx.dim) // 1024,), (1024,), (
                tensor_to_cparray(psps),
                tensor_to_cparray(grad_out.contiguous()),
                tensor_to_cparray(grad_x),
                cp.float32(ctx.tau),
                cp.float32(ctx.v_th),
                cp.int32(ctx.batch), 
                cp.int32(ctx.T), 
                cp.int32(ctx.dim),
                cp.int32(ctx.suro), 
                cp.float32(ctx.alpha))
            )
        return grad_x, None, None, None, None, None
