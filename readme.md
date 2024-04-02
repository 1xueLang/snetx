# snetx: Python package for spiking neural network

>A lighting python package for quickly building STBP-based spiking neural network.

## Install:

### Dependencies:

>PyTorch is needed to perform calculations. And GPU version is not necessary if hardware acceleration is not required. Appropriate PyTorch installation can be found at [PyTorch Get Start](https://pytorch.org/get-started/locally/).

>CuPy accelerated neuron model is provided in this package, and it is much faster then normal torch-based neuron model. Install appropriate CuPy is needed if you want to use CuPy accelerated neuron model. For example, for CUDA 10.2, use commend

```
pip install cupy-cuda102
```
>More versions can be found at [CuPy Install](https://docs.cupy.dev/en/stable/install.html#installing-cupy). Also, CuPy is not necessary.


### PyPi install

>This package can be installed from PyPi.
```
pip install snetx
```

## Test
```
cd test && python main.py
```

## Introduction

>One of the most significant differences between ***SNNs*** and ***ANNs*** is that the data in ***SNNs*** have an additional time dimension. 

>In this package, we treat the second dimension as the time dimension, so the data shape in this work are usually *N x T x D* or *N x T x C x H x W*.

>This package provides some basic components to conduct a spiking neural network, and some useful algorithms which can improve the performance of spiking neural networks. This package also provide common used data pre-processes, some auxiliary tools to help code the training process.

## Modules

### snetx.dataset.vision

>This package contains the functions to get **DataLoader** of the common used computer vision datasets. All functions have four parameters,

* data_dir: Root directory contains the dataset files.
* batch_size: Training batch size.
* test_batch_size: Testing batch size.
* transforms: Data transforms to perform data augments, set it to *None* to use the default transformations provided in package, which is suggested. And pass a list of [transforms_train, transforms_test] to customize transformations you need.
* return: train_data_loader: torch.utils.data.DataLoader, test_data_loader: torch.utils.data.DataLoader


>Following are functions provided.

#### MNIST 

>snetx.dataset.vision.mnist_dataset(data_dir, batch_size, test_batch_size, transforms=None)

#### FashionMNIST

>snetx.dataset.vision.Fmnist_dataset(data_dir, batch_size, test_batch_size, transforms=None)

#### CIFAR10

>snetx.dataset.vision.cifar10_dataset(data_dir, batch_size, test_size, transforms=None)

#### CIFAR100

>snetx.dataset.vision.cifar100_dataset(data_dir, batch_size, test_batch_size, transforms=None)

#### Usecase

```python
from snetx.dataset import vision

......

ds1, ds2 = vision.mnist_dataset('./datasets/', 32, 256)

for x, y in ds1:
    out = net(x.to(device))
    ......

```

### snetx.dataset.ddp.vision

>This package is the distributed data parallel version of **snetx.dataset.vision**, the function interface is the same, but use **torch.utils.data.distributed.DistributedSampler** to sample data. For ddp training, see [DDP](https://pytorch.org/docs/stable/notes/ddp.html).

### snetx.models.svggnet

>This package provide vggnets in ***SNNs***. Models from [Very Deep Convolutional Networks for Large-Scale Image Recognition](https://arxiv.org/abs/1409.1556)

#### snetx.models.svggnet.vgg11(**kwargs: Any)

#### snetx.models.svggnet.vgg11_bn(norm_layer, **kwargs: Any)

#### snetx.models.svggnet.vgg13(**kwargs: Any)

#### snetx.models.svggnet.vgg11_bn(norm_layer, **kwargs: Any)

#### snetx.models.svggnet.vgg16(**kwargs: Any)

#### snetx.models.svggnet.vgg11_bn(norm_layer, **kwargs: Any)

#### snetx.models.svggnet.vgg19(**kwargs: Any)

#### snetx.models.svggnet.vgg11_bn(norm_layer, **kwargs: Any)

#### Usecase

```python
from snetx.models import svggnet

net0 = svggnet.vgg16()
net1 = svggnet.vgg16_bn(nn.BatchNorm2d, num_classes=1000, dropout=0.5)
```

### snetx.snn.neuron

>This package contains spiking neruon models.

#### snetx.snn.neuron.LIF(tau: float = 2.0, v_th: float = 1.0, v_reset: float = 0.0, sg: torch.autograd.Function = snnalgo.arc_tan, alpha = lambda : 4.0, detach: bool = False)

* tau (float): Exponential attenuation coefficient of the membrane potential in LIF model. Default: 2.0.
* v_th (float): Threshold of membrane potential to emit spikes. Default: 1.0.
* v_reset (float): Reset membrane potential after spiking. Default: 0.0.
* sg (str): Surrogate function. Default: snetx.snn.algothrim.arc_tan.
* alpha (Callable): Parameter of surrogate function, which controls the function shape. Default: 4.0.
* detach (bool): Whether to keep the gradient trace after neuron spiking, *False* tp keep, default: *False*.

#### Usecase

```python
import snetx

sn = snetx.snn.neuron.LIF()
```

### snetx.snn.algorithm


#### snetx.snn.algothrim.Tosnn(model)

>As input in ***SNNs*** have an additional temporal dimension, so most modules in **torch.nn** can not process spiking input directly. **Tosnn** will convert any module in **torch.nn** to a module that can process spiking data.

#### Usecase

```python
import torch.nn as nn
import snetx.snn.algothrim as snnalgo

ann_x = torch.rand(32, 3, 32, 32)
ann_c = nn.Conv2d(3, 32, 3, 1, 1)
print(ann_c(ann_x).shape) # 32, 32, 32, 32

snn_x = torch.rand(32, 4, 3, 32, 32)
snn_c = snnalgo.Tosnn(nn.Conv2d(3, 32, 3, 1, 1))
print(snn_c(snn_x).shape) # 32, 4, 32, 32, 32

```

#### snetx.snn.algothrim.TETLoss(criterion, target, output, lamb=1e-3)

>Implementation of [Temporal Efficient Training of Spiking Neural Network via Gradient Re-weighting](https://arxiv.org/abs/2202.11946)

* criterion: Loss function.
* target: Labels like in ***ANNs***.
* output: Output of a spiking neural network. has a temporal dimension.
* lambda: default: 1e-3.

#### Usecase
```python
c = torch.nn.CrossEntropyLoss()

for x, y in ds:
    out = net(x)
    loss = snetx.snn.algothrim.TETLoss(c, y, out)
```

#### snetx.snn.algothrim.TET(criterion, lamb=1e-3)

#### Usecase
```python

criterion = snetx.snn.algothrim.TET(torch.nn.CrossEntropyLoss())

for x, y in ds:
    out = net(x)
    loss = criterion(y, out)
```

#### snetx.snn.algothrim.DirectEncoder(T)

>Repeat the input data for T times to encode the input data to a time series.

#### Usecase
```python
encoder = snetx.snn.algothrim.DirectEncoder(4)

for x, y in ds:
    x = encoder(x)
    ......
```

#### snetx.snn.algothrim.EvolvedAlpha(base)

>Many works like [IM-Loss: Information Maximization Loss for Spiking Neural Networks](https://proceedings.neurips.cc/paper_files/paper/2022/hash/010c5ba0cafc743fece8be02e7adb8dd-Abstract-Conference.html) demonstrate that with a gradually approximate surrogate function, spiking neural networks can be trained faster at the begining and achieve better performance at the end of the training.

>This module gradually increment the parameter which controls surrogate functions' shape to Implement this requirement.

#### Usecase

```python
import snetx.snn.algothrim as snnalgo

snnalgo.EvolvedAlpha.epoch = epoch
snnalgo.EvolvedAlpha.e_max = 1.

sn = snetx.snn.neuron.LIF(alpha=snalgo.EvolvedAlpha(base=4.))

for e in range(epoch):
    
    ......

    snnalgo.EvolvedAlpha.step()
```

>alpha = self.base * (1. + EvolvedAlpha.e_max * EvolvedAlpha.current / EvolvedAlpha.epoch), and EvolvedAlpha.step() execuate current = current + 1.

### snetx.cuend

#### snetx.cuend.neuron.LIF(tau: float = 2.0, v_th: float = 1.0, v_reset: float = 0.0, sg: str = 'arctan', alpha = lambda : 4.0, detach: bool = False)

>CuPy accelerated LIF model, it has the same function as snetx.snn.neuron.LIF but has faster execution speed.

* tau (float): Exponential attenuation coefficient of the membrane potential in LIF model. Default: 2.0.
* v_th (float): Threshold of membrane potential to emit spikes. Default: 1.0.
* v_reset (float): Reset membrane potential after spiking. Default: 0.0.
* sg (str):  Surrogate function index, see REGISTERED_SG_FUNCLIST. Default: 'arctan'.
* alpha (Callable): Parameter of surrogate function, which controls the function shape. Default: 4.0.
* detach (bool): Whether to keep the gradient trace after neuron spiking, *False* tp keep, default: *False*.

#### Usecase

```python
import snetx

sn = snetx.cuend.neuron.LIF()