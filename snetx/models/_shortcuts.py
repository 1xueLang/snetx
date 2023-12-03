from typing import Any, Callable, List, Optional

import torch.nn as nn
from torch import Tensor

# from ..transforms._presets import ImageClassification
# from ..utils import _log_api_usage_once
# from ._api import register_model, Weights, WeightsEnum
# from ._meta import _IMAGENET_CATEGORIES
# from ._utils import _ovewrite_named_param, handle_legacy_interface

from ..snn import algorithm as snnalgo


def conv3x3(in_planes: int, out_planes: int, stride: int = 1, groups: int = 1, dilation: int = 1) -> nn.Conv2d:
    """3x3 convolution with padding"""
    return nn.Conv2d(
        in_planes,
        out_planes,
        kernel_size=3,
        stride=stride,
        padding=dilation,
        groups=groups,
        bias=False,
        dilation=dilation,
    )

def conv3x3_norm(
    in_planes: int, out_planes: int, stride: int = 1, groups: int = 1, dilation: int = 1, norm_layer=None
    ) -> nn.Sequential:
    """3x3 convolution with padding"""
    layers = [snnalgo.Tosnn(conv3x3(in_planes, out_planes, stride, groups, dilation))]
    if norm_layer:
        layers.append(norm_layer(out_planes))
    
    return nn.Sequential(*layers)

def conv1x1(in_planes: int, out_planes: int, stride: int = 1) -> nn.Conv2d:
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)

def conv1x1_norm(in_planes: int, out_planes: int, stride: int = 1, norm_layer=None):
    """1x1 convolution"""
    layers = [snnalgo.Tosnn(conv1x1(in_planes, out_planes, stride=stride))]
    if norm_layer:
        layers.append(norm_layer(out_planes))
    
    return nn.Sequential(*layers)

# A
class BasicBlock_A(nn.Module):
    expansion: int = 1

    def __init__(
        self,
        neuron,
        n_config,
        inplanes: int,
        planes: int,
        stride: int = 1,
        downsample: Optional[nn.Module] = None,
        groups: int = 1,
        base_width: int = 64,
        dilation: int = 1,
        norm_layer: Optional[Callable[..., nn.Module]] = None,
    ) -> None:
        super().__init__()
        if norm_layer is None:
            norm_layer = snnalgo.tdBN2d
            # norm_layer = nn.BatchNorm2d
        if groups != 1 or base_width != 64:
            raise ValueError("BasicBlock only supports groups=1 and base_width=64")
        if dilation > 1:
            raise NotImplementedError("Dilation > 1 not supported in BasicBlock")
        # Both self.conv1 and self.downsample layers downsample the input when stride != 1
        self.sn1 = neuron(**n_config)
        self.conv1 = conv3x3_norm(inplanes, planes, stride, norm_layer=norm_layer)
        self.sn2 = neuron(**n_config)
        self.conv2 = conv3x3_norm(planes, planes, norm_layer=norm_layer)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x: Tensor) -> Tensor:
        identity = x

        out = self.sn1(x)
        if self.downsample:
            identity = self.downsample(identity)
            
        out = self.conv1(out)
        out = self.sn2(out)
        out = self.conv2(out)

        out += identity

        return out


class Bottleneck_A(nn.Module):
    # Bottleneck in torchvision places the stride for downsampling at 3x3 convolution(self.conv2)
    # while original implementation places the stride at the first 1x1 convolution(self.conv1)
    # according to "Deep residual learning for image recognition" https://arxiv.org/abs/1512.03385.
    # This variant is also known as ResNet V1.5 and improves accuracy according to
    # https://ngc.nvidia.com/catalog/model-scripts/nvidia:resnet_50_v1_5_for_pytorch.

    expansion: int = 4

    def __init__(
        self,
        neuron,
        n_config,
        inplanes: int,
        planes: int,
        stride: int = 1,
        downsample: Optional[nn.Module] = None,
        groups: int = 1,
        base_width: int = 64,
        dilation: int = 1,
        norm_layer: Optional[Callable[..., nn.Module]] = None,
    ) -> None:
        super().__init__()
        if norm_layer is None:
            norm_layer = snnalgo.tdBN2d
            # norm_layer = nn.BatchNorm2d
        width = int(planes * (base_width / 64.0)) * groups
        # Both self.conv2 and self.downsample layers downsample the input when stride != 1
        self.sn1 = neuron(**n_config)
        self.conv1 = conv1x1_norm(inplanes, width, norm_layer=norm_layer)
        self.sn2 = neuron(**n_config)
        self.conv2 = conv3x3_norm(width, width, stride, groups, dilation, norm_layer=norm_layer)
        self.sn3 = neuron(**n_config)
        self.conv3 = conv1x1_norm(width, planes * self.expansion, norm_layer=norm_layer)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x: Tensor) -> Tensor:
        identity = x
        
        out = self.sn1(x)
        if self.downsample:
            identity = self.downsample(identity)

        out = self.conv1(out)
        out = self.sn1(out)

        out = self.conv2(out)
        out = self.sn2(out)

        out = self.conv3(out)

        out += identity

        return out

# B
class BasicBlock_B(nn.Module):
    expansion: int = 1

    def __init__(
        self,
        neuron,
        n_config,
        inplanes: int,
        planes: int,
        stride: int = 1,
        downsample: Optional[nn.Module] = None,
        groups: int = 1,
        base_width: int = 64,
        dilation: int = 1,
        norm_layer: Optional[Callable[..., nn.Module]] = None,
    ) -> None:
        super().__init__()
        if norm_layer is None:
            norm_layer = snnalgo.tdBN2d
            # norm_layer = nn.BatchNorm2d
        if groups != 1 or base_width != 64:
            raise ValueError("BasicBlock only supports groups=1 and base_width=64")
        if dilation > 1:
            raise NotImplementedError("Dilation > 1 not supported in BasicBlock")
        # Both self.conv1 and self.downsample layers downsample the input when stride != 1
        self.bn1 = norm_layer(inplanes)
        self.sn1 = neuron(**n_config)
        self.conv1 = conv3x3_norm(inplanes, planes, stride, norm_layer=norm_layer)
        self.sn2 = neuron(**n_config)
        self.conv2 = conv3x3(planes, planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x: Tensor) -> Tensor:
        identity = x

        out = self.bn1(x)
        out = self.sn1(out)
        if self.downsample:
            identity = self.downsample(identity)
            
        out = self.conv1(out)
        out = self.sn2(out)
        out = self.conv2(out)

        out += identity

        return out


class Bottleneck_B(nn.Module):
    # Bottleneck in torchvision places the stride for downsampling at 3x3 convolution(self.conv2)
    # while original implementation places the stride at the first 1x1 convolution(self.conv1)
    # according to "Deep residual learning for image recognition" https://arxiv.org/abs/1512.03385.
    # This variant is also known as ResNet V1.5 and improves accuracy according to
    # https://ngc.nvidia.com/catalog/model-scripts/nvidia:resnet_50_v1_5_for_pytorch.

    expansion: int = 4

    def __init__(
        self,
        neuron,
        n_config,
        inplanes: int,
        planes: int,
        stride: int = 1,
        downsample: Optional[nn.Module] = None,
        groups: int = 1,
        base_width: int = 64,
        dilation: int = 1,
        norm_layer: Optional[Callable[..., nn.Module]] = None,
    ) -> None:
        super().__init__()
        if norm_layer is None:
            norm_layer = snnalgo.tdBN2d
            # norm_layer = nn.BatchNorm2d
        width = int(planes * (base_width / 64.0)) * groups
        # Both self.conv2 and self.downsample layers downsample the input when stride != 1
        self.bn1 = norm_layer(inplanes)
        self.sn1 = neuron(**n_config)
        self.conv1 = conv1x1_norm(inplanes, width, norm_layer=norm_layer)
        self.sn2 = neuron(**n_config)
        self.conv2 = conv3x3_norm(width, width, stride, groups, dilation, norm_layer=norm_layer)
        self.sn3 = neuron(**n_config)
        self.conv3 = conv1x1(width, planes * self.expansion)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x: Tensor) -> Tensor:
        identity = x

        out = self.bn1(x)
        out = self.sn1(out)
        if self.downsample:
            identity = self.downsample(identity)

        out = self.conv1(out)
        out = self.sn2(out)

        out = self.conv2(out)
        out = self.sn3(out)
        
        out = self.conv3(out)

        out += identity
        return out

# C
class BasicBlock_C(nn.Module):
    expansion: int = 1

    def __init__(
        self,
        neuron,
        n_config,
        inplanes: int,
        planes: int,
        stride: int = 1,
        downsample: Optional[nn.Module] = None,
        groups: int = 1,
        base_width: int = 64,
        dilation: int = 1,
        norm_layer: Optional[Callable[..., nn.Module]] = None,
    ) -> None:
        super().__init__()
        if norm_layer is None:
            norm_layer = snnalgo.tdBN2d
            # norm_layer = nn.BatchNorm2d
        if groups != 1 or base_width != 64:
            raise ValueError("BasicBlock only supports groups=1 and base_width=64")
        if dilation > 1:
            raise NotImplementedError("Dilation > 1 not supported in BasicBlock")
        # Both self.conv1 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv3x3_norm(inplanes, planes, stride, norm_layer=norm_layer)
        self.sn1 = neuron(**n_config)
        self.conv2 = conv3x3_norm(planes, planes, norm_layer=norm_layer)
        self.sn2 =neuron(**n_config)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x: Tensor) -> Tensor:
        identity = x
            
        out = self.conv1(out)
        out = self.sn1(out)
        out = self.conv2(out)
        
        if self.downsample:
            identity = self.downsample(identity)
        
        out += identity
        out = self.sn2(out)

        return out


class Bottleneck_C(nn.Module):
    # Bottleneck in torchvision places the stride for downsampling at 3x3 convolution(self.conv2)
    # while original implementation places the stride at the first 1x1 convolution(self.conv1)
    # according to "Deep residual learning for image recognition" https://arxiv.org/abs/1512.03385.
    # This variant is also known as ResNet V1.5 and improves accuracy according to
    # https://ngc.nvidia.com/catalog/model-scripts/nvidia:resnet_50_v1_5_for_pytorch.

    expansion: int = 4

    def __init__(
        self,
        neuron,
        n_config,
        inplanes: int,
        planes: int,
        stride: int = 1,
        downsample: Optional[nn.Module] = None,
        groups: int = 1,
        base_width: int = 64,
        dilation: int = 1,
        norm_layer: Optional[Callable[..., nn.Module]] = None,
    ) -> None:
        super().__init__()
        if norm_layer is None:
            norm_layer = snnalgo.tdBN2d
            # norm_layer = nn.BatchNorm2d
        width = int(planes * (base_width / 64.0)) * groups
        # Both self.conv2 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv1x1_norm(inplanes, width, norm_layer=norm_layer)
        self.sn1 = neuron(**n_config)
        self.conv2 = conv3x3_norm(width, width, stride, groups, dilation, norm_layer=norm_layer)
        self.sn2 = neuron(**n_config)
        self.conv3 = conv1x1_norm(width, planes * self.expansion, norm_layer=norm_layer)
        self.sn3 = neuron(**n_config)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x: Tensor) -> Tensor:
        identity = x

        out = self.conv1(x)
        out = self.sn1(out)

        out = self.conv2(out)
        out = self.sn2(out)

        out = self.conv3(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.sn3(out)

        return out

