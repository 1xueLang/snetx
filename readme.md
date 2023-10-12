# snetx: Python package for spiking neural network

>A lighting python package for quickly building STBP-based spiking neural network.

## Install:

### Dependencies:

>CuPy: Install the CuPy package of the corresponding version. For example, for CUDA 10.2, use commend

```
pip install cupy-cuda102
```
>More versions can be found at [CuPy Install](https://docs.cupy.dev/en/stable/install.html#installing-cupy).


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

#### snetx.snn.neuron.LIF(tau: float = 2.0, v_th: float = 1.0, v_reset: float = 0.0, sg: str = 'arctan', alpha = lambda : 4.0, detach: bool = False)

* tau (float): Exponential attenuation coefficient of the membrane potential in LIF model. Default: 2.0.
* v_th (float): Threshold of membrane potential to emit spikes. Default: 1.0.
* v_reset (float): Reset membrane potential after spiking. Default: 0.0.
* sg (str): Surrogate function index, see REGISTERED_SG_FUNCLIST. Default: 'arctan'.
* alpha (Callable): Parameter of surrogate function, which controls the function shape. Default: 4.0.
* detach (bool): Whether to keep the gradient trace after neuron spiking, *False* tp keep, default: *False*.

#### Usecase

```python
import snetx

sn = snetx.snn.neuron.LIF()
```
