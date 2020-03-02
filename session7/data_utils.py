from __future__ import print_function
import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F
from torchsummary import summary
import torch.optim as optim
from torchvision import datasets, transforms
from torch.optim.lr_scheduler import StepLR

import numpy as np
import matplotlib.pyplot as plt

# Check that GPU is avaiable

def get_device():
    # CUDA?
    cuda = torch.cuda.is_available()
    print("CUDA Available?", cuda)
    device = torch.device("cuda:0" if cuda else "cpu")
    print(device)
    return device
	
def get_data_transform():
    # Train Phase transformations
    train_transforms = transforms.Compose([
                                          #  transforms.Resize((28, 28)),
                                          #  transforms.ColorJitter(brightness=0.10, contrast=0.1, saturation=0.10, hue=0.1),
                                          #transforms.RandomRotation((-7.0, 7.0), fill=(1,)),
                                          transforms.ToTensor(),
                                          transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)) # The mean and std have to be sequences (e.g., tuples), therefore you should add a comma after the values. 
                                          # Note the difference between (0.1307) and (0.1307,)
                                          ])

    # Test Phase transformations
    test_transforms = transforms.Compose([
                                          #  transforms.Resize((28, 28)),
                                          #  transforms.ColorJitter(brightness=0.10, contrast=0.1, saturation=0.10, hue=0.1),
                                          transforms.ToTensor(),
                                          transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
                                          ])
    return train_transforms, test_transforms

def get_dataset(train_transforms, test_transforms):
    trainset = datasets.CIFAR10('./data', train=True, download=True, transform=train_transforms)
    testset = datasets.CIFAR10('./data', train=False, download=True, transform=test_transforms)
    return trainset, testset

def get_dataloader(batch_size, num_workers):

    cuda = torch.cuda.is_available()
    print("CUDA Available?", cuda)

    # dataloader arguments - something you'll fetch these from cmdprmt
    dataloader_args = dict(shuffle=True, batch_size=batch_size, num_workers=4, pin_memory=True) if cuda else dict(shuffle=True, batch_size=8)

    train_transforms, test_transforms = get_data_transform()

    trainset, testset = get_dataset(train_transforms, test_transforms)

    # train dataloader
    train_loader = torch.utils.data.DataLoader(trainset, **dataloader_args)

    # test dataloader
    test_loader = torch.utils.data.DataLoader(testset, **dataloader_args)

    return train_loader, test_loader
