import os
import torch
import torchvision

from spikingjelly.datasets.cifar10_dvs import CIFAR10DVS
from spikingjelly.datasets.dvs128_gesture import DVS128Gesture
from . import dvs_transform

from snetx import utils

def mnist_dataset(data_dir, batch_size, test_batch_size, transforms=None):
    if transforms == None:
        transform_train = torchvision.transforms.Compose([
            torchvision.transforms.RandomAffine(degrees=30, translate=(0.15, 0.15), scale=(0.85, 1.11)),
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize(0.1307, 0.3081),
        ])

        transform_test = torchvision.transforms.Compose([
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize(0.1307, 0.3081),
        ])
    else:
        transform_train, transform_test = transforms
        
    train_data_loader = torch.utils.data.DataLoader(
        dataset=torchvision.datasets.MNIST(
            root=os.path.join(data_dir, 'MNIST'),
            train=True,
            transform=transform_train,
            download=True
        ),
        batch_size=batch_size,
        shuffle=True,
        num_workers=4,
        drop_last=True,
        pin_memory=True
    )
    test_data_loader = torch.utils.data.DataLoader(
        dataset=torchvision.datasets.MNIST(
            root=os.path.join(data_dir, 'MNIST'),
            train=False,
            transform=transform_test,
            download=True
        ),
        batch_size=test_batch_size,
        shuffle=False,
        num_workers=4,
        drop_last=False,
        pin_memory=True
    )
    return train_data_loader, test_data_loader

def Fmnist_dataset(data_dir, batch_size, test_batch_size, transforms=None):
    if transforms == None:
        transform_train = torchvision.transforms.Compose([
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize(0.2860, 0.3530),
        ])
        transform_test = torchvision.transforms.Compose([
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize(0.2860, 0.3530),
        ])
    else:
        transform_train, transform_test = transforms
        
    train_data_loader = torch.utils.data.DataLoader(
        dataset=torchvision.datasets.FashionMNIST(
            root=os.path.join(data_dir, 'FashionMNIST'),
            train=True,
            transform=transform_train,
            download=True
        ),
        batch_size=batch_size,
        shuffle=True,
        num_workers=4,
        drop_last=True,
        pin_memory=True
    )
    test_data_loader = torch.utils.data.DataLoader(
        dataset=torchvision.datasets.FashionMNIST(
            root=os.path.join(data_dir, 'FashionMNIST'),
            train=False,
            transform=transform_test,
            download=True
        ),
        batch_size=test_batch_size,
        shuffle=False,
        num_workers=4,
        drop_last=False,
        pin_memory=True
    )
    return train_data_loader, test_data_loader

########### real time

def cifar10_dataset(data_dir, batch_size, test_size, transforms=None, auto=False):
    if transforms == None:
        transform_train = torchvision.transforms.Compose([
            # transforms.RandomResizedCrop(32, scale=(0.75,1.0), interpolation=PIL.Image.BILINEAR),
            torchvision.transforms.RandomCrop(32, padding=4),
            torchvision.transforms.RandomHorizontalFlip(),
            torchvision.transforms.AutoAugment(),
            torchvision.transforms.ToTensor(),
            utils.Cutout(1, 16),
            torchvision.transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ] if auto else [
            # transforms.RandomResizedCrop(32, scale=(0.75,1.0), interpolation=PIL.Image.BILINEAR),
            torchvision.transforms.RandomCrop(32, padding=4),
            torchvision.transforms.RandomHorizontalFlip(),
            # transforms.AutoAugment(),
            torchvision.transforms.ToTensor(),
            utils.Cutout(1, 16),
            torchvision.transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])

        transform_test = torchvision.transforms.Compose([
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])
    else:
        transform_train, transform_test = transforms
        
    train_data_loader = torch.utils.data.DataLoader(
        dataset=torchvision.datasets.CIFAR10(
            root=os.path.join(data_dir, 'CIFAR10'),
            train=True,
            transform=transform_train,
            download=True
        ),
        batch_size=batch_size,
        shuffle=True,
        num_workers=2,
        drop_last=True,
        pin_memory=True
    )
    test_data_loader = torch.utils.data.DataLoader(
        dataset=torchvision.datasets.CIFAR10(
            root=os.path.join(data_dir, 'CIFAR10'),
            train=False,
            transform=transform_test,
            download=True
        ),
        batch_size=test_size,
        shuffle=False,
        num_workers=2,
        drop_last=False,
        pin_memory=True
    )
    return train_data_loader, test_data_loader

def cifar100_dataset(data_dir, batch_size, test_batch_size, transforms=None):
    if transforms == None:
        normalize = torchvision.transforms.Normalize(
            mean=[0.507, 0.487, 0.441],
            std=[0.267, 0.256, 0.276],
        )
        transform_train = torchvision.transforms.Compose([
            torchvision.transforms.RandomCrop(32, padding=4),
            torchvision.transforms.RandomHorizontalFlip(0.5),
            torchvision.transforms.ToTensor(),
            utils.Cutout(1, 16),
            normalize
        ])
        
        transform_test = torchvision.transforms.Compose([
        #     transforms.RandomCrop(32, padding=4),
        #     transforms.RandomHorizontalFlip(0.5),
            torchvision.transforms.ToTensor(),
            normalize
        ])
    else:
        transform_train, transform_test = transforms
    
    train_data_loader = torch.utils.data.DataLoader(
        dataset=torchvision.datasets.CIFAR100(
            root=os.path.join(data_dir, 'CIFAR100'),
            train=True,
            transform=transform_train,
            download=True
        ),
        batch_size=batch_size,
        shuffle=False,
        num_workers=2,
        drop_last=False,
        pin_memory=True
    )
    
    test_data_loader = torch.utils.data.DataLoader(
        dataset=torchvision.datasets.CIFAR100(
            root=os.path.join(data_dir, 'CIFAR100'),
            train=False,
            transform=transform_test,
            download=True
        ),
        batch_size=test_batch_size,
        shuffle=False,
        num_workers=2,
        drop_last=False,
        pin_memory=True
    )
    return train_data_loader, test_data_loader


def imagenet_dataset(data_dir, batch_size, test_batch_size, transform=None):
    if transform:
        transform_train, transform_test = transform
    else:
        transform_train = torchvision.transforms.Compose([
            torchvision.transforms.RandomResizedCrop(224),
            torchvision.transforms.AutoAugment(),
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
        transform_test = torchvision.transforms.Compose([
            torchvision.transforms.Resize(256),
            torchvision.transforms.CenterCrop(224),
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
    tr_set = torchvision.datasets.ImageFolder(
        os.path.join(data_dir, 'train'),
        transform=transform_train
    )
    ts_set = torchvision.datasets.ImageFolder(
        os.path.join(data_dir, 'val'),
        transform=transform_test
    )
    
    return torch.utils.data.DataLoader(tr_set, shuffle=True, num_workers=16, batch_size=batch_size, pin_memory=True), \
        torch.utils.data.DataLoader(ts_set, shuffle=False, num_workers=16, batch_size=test_batch_size, pin_memory=True)


########### DVS

def cifar10dvs_dataset(data_dir, batch_size1, batch_size2, T):
    transform_train = torchvision.transforms.Compose([
        dvs_transform.ToTensor(),
        torchvision.transforms.Resize(size=(48, 48)),
        torchvision.transforms.RandomCrop(48, padding=4),
        torchvision.transforms.RandomHorizontalFlip(),])
        # transforms.Normalize(mean=[n / 255. for n in [129.3, 124.1, 112.4]],std=[n / 255. for n in [68.2, 65.4, 70.4]]),
        # Cutout(n_holes=1, length=16)])

    transform_test = torchvision.transforms.Compose([
        dvs_transform.ToTensor(),
        torchvision.transforms.Resize(size=(48, 48))])
        # transforms.Normalize(mean=[n / 255. for n in [129.3, 124.1, 112.4]], std=[n / 255. for n in [68.2, 65.4, 70.4]])
    trainset = CIFAR10DVS(root=os.path.join(data_dir, 'CIFAR10DVS'), train=True, data_type='frame', frames_number=T, split_by='number', transform=transform_train)
    testset = CIFAR10DVS(root=os.path.join(data_dir, 'CIFAR10DVS'), train=False, data_type='frame', frames_number=T, split_by='number',  transform=transform_test)
    train_data_loader = torch.utils.data.DataLoader(trainset, batch_size=batch_size1, shuffle=True, num_workers=2)
    test_data_loader = torch.utils.data.DataLoader(testset, batch_size=batch_size2, shuffle=False, num_workers=2)
    return train_data_loader, test_data_loader


def dvs128gesture_dataset(data_dir, batch_size1, batch_size2, T):
    trainset = DVS128Gesture(data_dir, train=True, data_type='frame', frames_number=T, split_by='number',)
                             
    train_data_loader = torch.utils.data.DataLoader(trainset, batch_size=batch_size1, shuffle=True, num_workers=2)
    
    testset = DVS128Gesture(data_dir, train=False, data_type='frame', frames_number=T, split_by='number')
    test_data_loader = torch.utils.data.DataLoader(testset, batch_size=batch_size2, shuffle=False, num_workers=2)
    return train_data_loader, test_data_loader