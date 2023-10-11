import torchvision, torch, os
from torchvision import transforms
import PIL

from snetx import utils

def mnist_dataset(data_dir, batch_size, test_batch_size):
    transform_train = transforms.Compose([
        transforms.RandomAffine(degrees=30, translate=(0.15, 0.15), scale=(0.85, 1.11)),
        transforms.ToTensor(),
        transforms.Normalize(0.1307, 0.3081),
#         torch.flatten
    ])

    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(0.1307, 0.3081),
#         torch.flatten
    ])
    
    train_set = torchvision.datasets.MNIST(
        root=os.path.join(data_dir, 'MNIST'),
        train=True,
        transform=transform_train,
        download=True
    )
    train_sampler = torch.utils.data.distributed.DistributedSampler(train_set)
    
    train_data_loader = torch.utils.data.DataLoader(
        dataset=train_set,
        batch_size=batch_size,
        # shuffle=True,
        sampler=train_sampler,
        num_workers=4,
        drop_last=True,
        pin_memory=True
    )
    
    test_set = torchvision.datasets.MNIST(
        root=os.path.join(data_dir, 'MNIST'),
        train=False,
        transform=transform_test,
        download=True
    )
    test_sampler = torch.utils.data.distributed.DistributedSampler(test_set)
    
    test_data_loader = torch.utils.data.DataLoader(
        dataset=test_set,    
        batch_size=test_batch_size,
        # sampler=test_sampler,
        num_workers=4,
        drop_last=False,
        pin_memory=True
    )
    return train_data_loader, test_data_loader

def Fmnist_dataset(data_dir, batch_size, test_batch_size):
    transform_train = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(0.2860, 0.3530),
    ])
    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(0.2860, 0.3530),
    ])
    train_set = torchvision.datasets.FashionMNIST(
        root=os.path.join(data_dir, 'FashionMNIST'),
        train=True,
        transform=transform_train,
        download=True
    )
    train_sampler = torch.utils.data.distributed.DistributedSampler(train_set)
    
    train_data_loader = torch.utils.data.DataLoader(
        dataset=train_set,
        batch_size=batch_size,
        # shuffle=True,
        sampler=train_sampler,
        num_workers=4,
        drop_last=True,
        pin_memory=True
    )
    
    test_set = torchvision.datasets.FashionMNIST(
        root=os.path.join(data_dir, 'FashionMNIST'),
        train=False,
        transform=transform_test,
        download=True
    )
    test_sampler = torch.utils.data.distributed.DistributedSampler(test_set)
    test_data_loader = torch.utils.data.DataLoader(
        dataset=test_set,
        batch_size=test_batch_size,
        # sampler=test_sampler,
        num_workers=4,
        drop_last=False,
        pin_memory=True
    )
    return train_data_loader, test_data_loader

def cifar10_dataset(data_dir, batch_size, test_size):
    transform_train = transforms.Compose([
        # transforms.RandomResizedCrop(32, scale=(0.75,1.0), interpolation=PIL.Image.BILINEAR),
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        # transforms.AutoAugment(),
        transforms.ToTensor(),
        utils.Cutout(1, 16),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])
    train_set = torchvision.datasets.CIFAR10(
        root=os.path.join(data_dir, 'CIFAR10'),
        train=True,
        transform=transform_train,
        download=True
    )
    train_sampler = torch.utils.data.distributed.DistributedSampler(train_set)
    train_data_loader = torch.utils.data.DataLoader(
        dataset=train_set,
        batch_size=batch_size,
        # shuffle=True,
        sampler=train_sampler,
        num_workers=4,
        drop_last=True,
        pin_memory=True
    )
    test_set = torchvision.datasets.CIFAR10(
        root=os.path.join(data_dir, 'CIFAR10'),
        train=False,
        transform=transform_test,
        download=True
    )
    test_sampler = torch.utils.data.distributed.DistributedSampler(test_set)
    test_data_loader = torch.utils.data.DataLoader(
        dataset=test_set,
        batch_size=test_size,
        # sampler=test_sampler,
        num_workers=4,
        drop_last=False,
        pin_memory=True)
    return train_data_loader, test_data_loader

def cifar100_dataset(data_dir, batch_size, test_batch_size):
    normalize = transforms.Normalize(mean=[0.507, 0.487, 0.441],
                                     std=[0.267, 0.256, 0.276])
    
    train_transform = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(0.5),
        transforms.ToTensor(),
        normalize
    ])
    
    test_transform = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(0.5),
        transforms.ToTensor(),
        normalize
    ])
    
    train_set = torchvision.datasets.CIFAR100(
        root=data_dir + 'CIFAR100',
        train=True,
        transform=train_transform,
        download=True
    )
    train_sampler = torch.utils.data.distributed.DistributedSampler(train_set)
    
    train_data_loader = torch.utils.data.DataLoader(
        dataset=train_set,
        batch_size=batch_size,
        # shuffle=True,
        sampler=train_sampler,
        num_workers=2,
        drop_last=False,
        pin_memory=True
    )
    test_set = torchvision.datasets.CIFAR100(
        root=data_dir + 'CIFAR100',
        train=False,
        transform=train_transform,
        download=True
    )
    test_sampler = torch.utils.data.distributed.DistributedSampler(test_set)
    
    test_data_loader = torch.utils.data.DataLoader(
        dataset=test_set,
        batch_size=test_batch_size,
        # shuffle=False,
        # sampler=test_sampler,
        num_workers=2,
        drop_last=False,
        pin_memory=True
    )
    return train_data_loader, test_data_loader