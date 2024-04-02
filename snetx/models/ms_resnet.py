from functools import partial
from typing import Any, Callable, List, Optional, Dict, TypeVar

import torch
import torch.nn as nn
from torch import Tensor

# from ..transforms._presets import ImageClassification
# from ..utils import _log_api_usage_once
# from ._api import register_model, Weights, WeightsEnum
# from ._meta import _IMAGENET_CATEGORIES
# from ._utils import _ovewrite_named_param, handle_legacy_interface

from ..snn import algorithm as snnalgo
from ._shortcuts import BasicBlock_A, BasicBlock_B, BasicBlock_C
from ._shortcuts import Bottleneck_A, Bottleneck_B, Bottleneck_C
from ._shortcuts import conv1x1, conv1x1_norm


__all__ = [
    "ResNet",
    # "ResNet18_Weights",
    # "ResNet34_Weights",
    # "ResNet50_Weights",
    # "ResNet101_Weights",
    # "ResNet152_Weights",
    # "ResNeXt50_32X4D_Weights",
    # "ResNeXt101_32X8D_Weights",
    # "ResNeXt101_64X4D_Weights",
    # "Wide_ResNet50_2_Weights",
    # "Wide_ResNet101_2_Weights",
    "resnet18",
    "resnet34",
    "resnet50",
    "resnet101",
    "resnet152",
    "resnext50_32x4d",
    "resnext101_32x8d",
    "resnext101_64x4d",
    "wide_resnet50_2",
    "wide_resnet101_2",
]

BasicBlock = {
    'A': BasicBlock_A,
    'B': BasicBlock_B,
    'C': BasicBlock_C
}

Bottleneck = {
    'A': Bottleneck_A,
    'B': Bottleneck_B,
    'C': Bottleneck_C,
}

def cifar10_feature(type, norm_layer, inplanes, neuron, n_config):
    layers = [snnalgo.Tosnn(nn.Conv2d(3, inplanes, kernel_size=3, stride=1, padding=1, bias=False)),]
    if type == 'A' or type == 'C':
        layers.append(norm_layer(inplanes))
    
    if type == 'C':
        layers.append(neuron(**n_config))
    
    layers = nn.Sequential(*layers)
    
    return layers

def cifar10dvs_feature(type, norm_layer, inplanes, neuron, n_config):
    layers = [snnalgo.Tosnn(nn.Conv2d(2, inplanes, kernel_size=3, stride=1, padding=1, bias=False)),]
    if type == 'A' or type == 'C':
        layers.append(norm_layer(inplanes))
    
    if type == 'C':
        layers.append(neuron(**n_config))
        
    layers = nn.Sequential(*layers)
    
    return layers

def classic_feature(type, norm_layer, inplanes, neuron, n_config):
    layers = [
        snnalgo.Tosnn(nn.Conv2d(3, inplanes, kernel_size=7, stride=2, padding=3)),
    ]
    if type == 'A' or type == 'C':
        layers.append(norm_layer(inplanes)),
    if type == 'C':
        layers.append(neuron(**n_config))
        
    layers.append(snnalgo.Tosnn(nn.MaxPool2d(kernel_size=3, stride=2, padding=1)))
    
    return nn.Sequential(*layers)

class ResNet(nn.Module):
    def __init__(
        self,
        neuron,
        n_config,
        block,
        layers: List[int],
        num_classes: int = 1000,
        zero_init_residual: bool = False,
        groups: int = 1,
        width_per_group: int = 64,
        replace_stride_with_dilation: Optional[List[bool]] = None,
        norm_layer: Optional[Callable[..., nn.Module]] = None,
        type: str = 'A',
        feature: Callable = classic_feature
    ) -> None:
        super().__init__()
        self.type = type
        if norm_layer is None:
            norm_layer = snnalgo.tdBN2d
            # norm_layer = nn.BatchNorm2d
        self._norm_layer = norm_layer

        self.inplanes = 64
        self.dilation = 1
        if replace_stride_with_dilation is None:
            # each element in the tuple indicates if we should replace
            # the 2x2 stride with a dilated convolution instead
            replace_stride_with_dilation = [False, False, False]
        if len(replace_stride_with_dilation) != 3:
            raise ValueError(
                "replace_stride_with_dilation should be None "
                f"or a 3-element tuple, got {replace_stride_with_dilation}"
            )
        self.groups = groups
        self.base_width = width_per_group
        self.feature = feature(self.type, norm_layer, self.inplanes, neuron, n_config)
        self.layer1 = self._make_layer(neuron, n_config, block, 64, layers[0])
        self.layer2 = self._make_layer(neuron, n_config, block, 128, layers[1], stride=2, dilate=replace_stride_with_dilation[0])
        self.layer3 = self._make_layer(neuron, n_config, block, 256, layers[2], stride=2, dilate=replace_stride_with_dilation[1])
        self.layer4 = self._make_layer(neuron, n_config, block, 512, layers[3], stride=2, dilate=replace_stride_with_dilation[2])
        
        out_layers = []
        if self.type == 'A':
            out_layers.extend([neuron(**n_config), snnalgo.Tosnn(nn.AdaptiveAvgPool2d((1, 1)))])
        elif self.type == 'B':
            out_layers.extend([norm_layer(512 * block.expansion), neuron(**n_config), snnalgo.Tosnn(nn.AdaptiveAvgPool2d((1, 1)))])
        else:
            out_layers.append(snnalgo.Tosnn(nn.AdaptiveAvgPool2d((1, 1))))
            
        self.out_layer = nn.Sequential(*out_layers)
        self.fc = nn.Linear(512 * block.expansion, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            # elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
            #     if m.affine:
            #         nn.init.constant_(m.weight, 1)
            #         nn.init.constant_(m.bias, 0)

        # Zero-initialize the last BN in each residual branch,
        # so that the residual branch starts with zeros, and each residual block behaves like an identity.
        # This improves the model by 0.2~0.3% according to https://arxiv.org/abs/1706.02677
        if zero_init_residual:
            for m in self.modules():
                if (isinstance(m, Bottleneck_A) or isinstance(m, Bottleneck_C)) and m.conv3[-1].weight is not None:
                    nn.init.constant_(m.conv3[-1].weight, 0)  # type: ignore[arg-type]
                elif (isinstance(m, BasicBlock_A) or isinstance(m, BasicBlock_C)) and m.conv2[-1].weight is not None:
                    nn.init.constant_(m.conv2[-1].weight, 0)  # type: ignore[arg-type]

    def _make_layer(
        self,
        neuron,
        n_config,
        block,
        planes: int,
        blocks: int,
        stride: int = 1,
        dilate: bool = False,
    ) -> nn.Sequential:
        norm_layer = self._norm_layer
        downsample = None
        previous_dilation = self.dilation
        if dilate:
            self.dilation *= stride
            stride = 1
        if stride != 1 or self.inplanes != planes * block.expansion:
            if self.type == 'A':
                downsample = nn.Sequential(
                    neuron(**n_config),
                    conv1x1_norm(self.inplanes, planes * block.expansion, stride, norm_layer=norm_layer)
                )
            elif self.type == 'B':
                if norm_layer:
                    downsample = [snnalgo.Tosnn(norm_layer(self.inplanes))]
                else:
                    downsample = []
                downsample += [neuron(**n_config), conv1x1(self.inplanes, planes * block.expansion, stride)]
                downsample = nn.Sequential(*downsample)
            elif self.type == 'C':
                downsample = conv1x1_norm(self.inplanes, planes * block.expansion, stride, norm_layer=norm_layer)
            else:
                raise ValueError('')

        layers = []
        layers.append(
            block(
                neuron, n_config, self.inplanes, planes, stride, downsample, self.groups, self.base_width, previous_dilation, norm_layer
            )
        )
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(
                block(
                    neuron,
                    n_config,
                    self.inplanes,
                    planes,
                    groups=self.groups,
                    base_width=self.base_width,
                    dilation=self.dilation,
                    norm_layer=norm_layer,
                )
            )

        return nn.Sequential(*layers)

    def _forward_impl(self, x: Tensor) -> Tensor:
        # See note [TorchScript super()]
        x = self.feature(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.out_layer(x)
        x = torch.flatten(x, 2)
        x = self.fc(x)

        return x

    def forward(self, x: Tensor) -> Tensor:
        return self._forward_impl(x)


def _resnet(
    type,
    neuron,
    n_config,
    block,
    layers: List[int],
    **kwargs: Any,
) -> ResNet:

    model = ResNet(neuron, n_config, block, layers, type=type, **kwargs)

    return model

V = TypeVar('V')

def _ovewrite_named_param(kwargs: Dict[str, Any], param: str, new_value: V) -> None:
    if param in kwargs:
        if kwargs[param] != new_value:
            raise ValueError(f"The parameter '{param}' expected value {new_value} but got {kwargs[param]} instead.")
    else:
        kwargs[param] = new_value


def resnet18(type, neuron, n_config, **kwargs: Any) -> ResNet:
    """ResNet-18 from `Deep Residual Learning for Image Recognition <https://arxiv.org/abs/1512.03385>`__.

    Args:
        weights (:class:`~torchvision.models.ResNet18_Weights`, optional): The
            pretrained weights to use. See
            :class:`~torchvision.models.ResNet18_Weights` below for
            more details, and possible values. By default, no pre-trained
            weights are used.
        progress (bool, optional): If True, displays a progress bar of the
            download to stderr. Default is True.
        **kwargs: parameters passed to the ``torchvision.models.resnet.ResNet``
            base class. Please refer to the `source code
            <https://github.com/pytorch/vision/blob/main/torchvision/models/resnet.py>`_
            for more details about this class.

    .. autoclass:: torchvision.models.ResNet18_Weights
        :members:
    """
    
    return _resnet(type, neuron, n_config, BasicBlock[type], [2, 2, 2, 2], **kwargs)


def resnet34(type, neuron, n_config, **kwargs: Any) -> ResNet:
    """ResNet-34 from `Deep Residual Learning for Image Recognition <https://arxiv.org/abs/1512.03385>`__.

    Args:
        weights (:class:`~torchvision.models.ResNet34_Weights`, optional): The
            pretrained weights to use. See
            :class:`~torchvision.models.ResNet34_Weights` below for
            more details, and possible values. By default, no pre-trained
            weights are used.
        progress (bool, optional): If True, displays a progress bar of the
            download to stderr. Default is True.
        **kwargs: parameters passed to the ``torchvision.models.resnet.ResNet``
            base class. Please refer to the `source code
            <https://github.com/pytorch/vision/blob/main/torchvision/models/resnet.py>`_
            for more details about this class.

    .. autoclass:: torchvision.models.ResNet34_Weights
        :members:
    """
    return _resnet(type, neuron, n_config, BasicBlock[type], [3, 4, 6, 3], **kwargs)


def resnet50(type, neuron, n_config, **kwargs: Any) -> ResNet:
    """ResNet-50 from `Deep Residual Learning for Image Recognition <https://arxiv.org/abs/1512.03385>`__.

    .. note::
       The bottleneck of TorchVision places the stride for downsampling to the second 3x3
       convolution while the original paper places it to the first 1x1 convolution.
       This variant improves the accuracy and is known as `ResNet V1.5
       <https://ngc.nvidia.com/catalog/model-scripts/nvidia:resnet_50_v1_5_for_pytorch>`_.

    Args:
        weights (:class:`~torchvision.models.ResNet50_Weights`, optional): The
            pretrained weights to use. See
            :class:`~torchvision.models.ResNet50_Weights` below for
            more details, and possible values. By default, no pre-trained
            weights are used.
        progress (bool, optional): If True, displays a progress bar of the
            download to stderr. Default is True.
        **kwargs: parameters passed to the ``torchvision.models.resnet.ResNet``
            base class. Please refer to the `source code
            <https://github.com/pytorch/vision/blob/main/torchvision/models/resnet.py>`_
            for more details about this class.

    .. autoclass:: torchvision.models.ResNet50_Weights
        :members:
    """

    return _resnet(type, neuron, n_config, Bottleneck[type], [3, 4, 6, 3], **kwargs)


def resnet101(type, neuron, n_config, **kwargs: Any) -> ResNet:
    """ResNet-101 from `Deep Residual Learning for Image Recognition <https://arxiv.org/abs/1512.03385>`__.

    .. note::
       The bottleneck of TorchVision places the stride for downsampling to the second 3x3
       convolution while the original paper places it to the first 1x1 convolution.
       This variant improves the accuracy and is known as `ResNet V1.5
       <https://ngc.nvidia.com/catalog/model-scripts/nvidia:resnet_50_v1_5_for_pytorch>`_.

    Args:
        weights (:class:`~torchvision.models.ResNet101_Weights`, optional): The
            pretrained weights to use. See
            :class:`~torchvision.models.ResNet101_Weights` below for
            more details, and possible values. By default, no pre-trained
            weights are used.
        progress (bool, optional): If True, displays a progress bar of the
            download to stderr. Default is True.
        **kwargs: parameters passed to the ``torchvision.models.resnet.ResNet``
            base class. Please refer to the `source code
            <https://github.com/pytorch/vision/blob/main/torchvision/models/resnet.py>`_
            for more details about this class.

    .. autoclass:: torchvision.models.ResNet101_Weights
        :members:
    """
    return _resnet(type, neuron, n_config, Bottleneck[type], [3, 4, 23, 3], **kwargs)


def resnet152(type, neuron, n_config, **kwargs: Any) -> ResNet:
    """ResNet-152 from `Deep Residual Learning for Image Recognition <https://arxiv.org/abs/1512.03385>`__.

    .. note::
       The bottleneck of TorchVision places the stride for downsampling to the second 3x3
       convolution while the original paper places it to the first 1x1 convolution.
       This variant improves the accuracy and is known as `ResNet V1.5
       <https://ngc.nvidia.com/catalog/model-scripts/nvidia:resnet_50_v1_5_for_pytorch>`_.

    Args:
        weights (:class:`~torchvision.models.ResNet152_Weights`, optional): The
            pretrained weights to use. See
            :class:`~torchvision.models.ResNet152_Weights` below for
            more details, and possible values. By default, no pre-trained
            weights are used.
        progress (bool, optional): If True, displays a progress bar of the
            download to stderr. Default is True.
        **kwargs: parameters passed to the ``torchvision.models.resnet.ResNet``
            base class. Please refer to the `source code
            <https://github.com/pytorch/vision/blob/main/torchvision/models/resnet.py>`_
            for more details about this class.

    .. autoclass:: torchvision.models.ResNet152_Weights
        :members:
    """

    return _resnet(type, neuron, n_config, Bottleneck[type], [3, 8, 36, 3], **kwargs)


def resnext50_32x4d(type, neuron, n_config, **kwargs: Any) -> ResNet:
    """ResNeXt-50 32x4d model from
    `Aggregated Residual Transformation for Deep Neural Networks <https://arxiv.org/abs/1611.05431>`_.

    Args:
        weights (:class:`~torchvision.models.ResNeXt50_32X4D_Weights`, optional): The
            pretrained weights to use. See
            :class:`~torchvision.models.ResNext50_32X4D_Weights` below for
            more details, and possible values. By default, no pre-trained
            weights are used.
        progress (bool, optional): If True, displays a progress bar of the
            download to stderr. Default is True.
        **kwargs: parameters passed to the ``torchvision.models.resnet.ResNet``
            base class. Please refer to the `source code
            <https://github.com/pytorch/vision/blob/main/torchvision/models/resnet.py>`_
            for more details about this class.
    .. autoclass:: torchvision.models.ResNeXt50_32X4D_Weights
        :members:
    """
    _ovewrite_named_param(kwargs, "groups", 32)
    _ovewrite_named_param(kwargs, "width_per_group", 4)
    return _resnet(type, neuron, n_config, Bottleneck[type], [3, 4, 6, 3], **kwargs)


def resnext101_32x8d(type, neuron, n_config, **kwargs: Any) -> ResNet:
    """ResNeXt-101 32x8d model from
    `Aggregated Residual Transformation for Deep Neural Networks <https://arxiv.org/abs/1611.05431>`_.

    Args:
        weights (:class:`~torchvision.models.ResNeXt101_32X8D_Weights`, optional): The
            pretrained weights to use. See
            :class:`~torchvision.models.ResNeXt101_32X8D_Weights` below for
            more details, and possible values. By default, no pre-trained
            weights are used.
        progress (bool, optional): If True, displays a progress bar of the
            download to stderr. Default is True.
        **kwargs: parameters passed to the ``torchvision.models.resnet.ResNet``
            base class. Please refer to the `source code
            <https://github.com/pytorch/vision/blob/main/torchvision/models/resnet.py>`_
            for more details about this class.
    .. autoclass:: torchvision.models.ResNeXt101_32X8D_Weights
        :members:
    """
    _ovewrite_named_param(kwargs, "groups", 32)
    _ovewrite_named_param(kwargs, "width_per_group", 8)
    return _resnet(type, neuron, n_config, Bottleneck[type], [3, 4, 23, 3], **kwargs)


def resnext101_64x4d(type, neuron, n_config, **kwargs: Any) -> ResNet:
    """ResNeXt-101 64x4d model from
    `Aggregated Residual Transformation for Deep Neural Networks <https://arxiv.org/abs/1611.05431>`_.

    Args:
        weights (:class:`~torchvision.models.ResNeXt101_64X4D_Weights`, optional): The
            pretrained weights to use. See
            :class:`~torchvision.models.ResNeXt101_64X4D_Weights` below for
            more details, and possible values. By default, no pre-trained
            weights are used.
        progress (bool, optional): If True, displays a progress bar of the
            download to stderr. Default is True.
        **kwargs: parameters passed to the ``torchvision.models.resnet.ResNet``
            base class. Please refer to the `source code
            <https://github.com/pytorch/vision/blob/main/torchvision/models/resnet.py>`_
            for more details about this class.
    .. autoclass:: torchvision.models.ResNeXt101_64X4D_Weights
        :members:
    """
    _ovewrite_named_param(kwargs, "groups", 64)
    _ovewrite_named_param(kwargs, "width_per_group", 4)
    return _resnet(type, neuron, n_config, Bottleneck[type], [3, 4, 23, 3], **kwargs)


def wide_resnet50_2(type, neuron, n_config, **kwargs: Any) -> ResNet:
    """Wide ResNet-50-2 model from
    `Wide Residual Networks <https://arxiv.org/abs/1605.07146>`_.

    The model is the same as ResNet except for the bottleneck number of channels
    which is twice larger in every block. The number of channels in outer 1x1
    convolutions is the same, e.g. last block in ResNet-50 has 2048-512-2048
    channels, and in Wide ResNet-50-2 has 2048-1024-2048.

    Args:
        weights (:class:`~torchvision.models.Wide_ResNet50_2_Weights`, optional): The
            pretrained weights to use. See
            :class:`~torchvision.models.Wide_ResNet50_2_Weights` below for
            more details, and possible values. By default, no pre-trained
            weights are used.
        progress (bool, optional): If True, displays a progress bar of the
            download to stderr. Default is True.
        **kwargs: parameters passed to the ``torchvision.models.resnet.ResNet``
            base class. Please refer to the `source code
            <https://github.com/pytorch/vision/blob/main/torchvision/models/resnet.py>`_
            for more details about this class.
    .. autoclass:: torchvision.models.Wide_ResNet50_2_Weights
        :members:
    """
    _ovewrite_named_param(kwargs, "width_per_group", 64 * 2)
    return _resnet(type, neuron, n_config, Bottleneck[type], [3, 4, 6, 3], **kwargs)


def wide_resnet101_2(type, neuron, n_config, **kwargs: Any) -> ResNet:
    """Wide ResNet-101-2 model from
    `Wide Residual Networks <https://arxiv.org/abs/1605.07146>`_.

    The model is the same as ResNet except for the bottleneck number of channels
    which is twice larger in every block. The number of channels in outer 1x1
    convolutions is the same, e.g. last block in ResNet-101 has 2048-512-2048
    channels, and in Wide ResNet-101-2 has 2048-1024-2048.

    Args:
        weights (:class:`~torchvision.models.Wide_ResNet101_2_Weights`, optional): The
            pretrained weights to use. See
            :class:`~torchvision.models.Wide_ResNet101_2_Weights` below for
            more details, and possible values. By default, no pre-trained
            weights are used.
        progress (bool, optional): If True, displays a progress bar of the
            download to stderr. Default is True.
        **kwargs: parameters passed to the ``torchvision.models.resnet.ResNet``
            base class. Please refer to the `source code
            <https://github.com/pytorch/vision/blob/main/torchvision/models/resnet.py>`_
            for more details about this class.
    .. autoclass:: torchvision.models.Wide_ResNet101_2_Weights
        :members:
    """
    _ovewrite_named_param(kwargs, "width_per_group", 64 * 2)
    return _resnet(type, neuron, n_config, Bottleneck[type], [3, 4, 23, 3], **kwargs)