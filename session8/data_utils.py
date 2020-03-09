from __future__ import print_function
import torch
import torchvision
from torchvision import datasets, transforms

import numpy as np


# Check that GPU is avaiable

def get_device():
    # CUDA?
    cuda = torch.cuda.is_available()
    print("CUDA Available?3", cuda)
    device = torch.device("cuda:0" if cuda else "cpu")
    print(device)
    return device
	
def calculate_dataset_mean_std():
    trainset = datasets.CIFAR10('./data', train=True, download=True, transform=transforms.ToTensor())
    testset = datasets.CIFAR10('./data', train=False, download=True, transform=transforms.ToTensor())

    data = np.concatenate([trainset.data, testset.data], axis=0)
    data = data.astype(np.float32)/255.

    print("\nTotal dataset(train+test) shape: ", data.shape)

    means = []
    stdevs = []
    for i in range(3): # 3 channels
        pixels = data[:,:,:,i].ravel()
        means.append(np.mean(pixels))
        stdevs.append(np.std(pixels))

    return (means[0], means[1], means[2]), (stdevs[0], stdevs[1], stdevs[2])

def get_dataset(train_transforms, test_transforms):
    trainset = datasets.CIFAR10('./data', train=True, download=True, transform=train_transforms)
    testset = datasets.CIFAR10('./data', train=False, download=True, transform=test_transforms)
    return trainset, testset

def get_dataloader(train_transforms, test_transforms, batch_size, num_workers):

    cuda = torch.cuda.is_available()
    print("CUDA Available?", cuda)

    # dataloader arguments - something you'll fetch these from cmdprmt
    dataloader_args = dict(shuffle=True, batch_size=batch_size, num_workers=4, pin_memory=True) if cuda else dict(shuffle=True, batch_size=8)

    trainset, testset = get_dataset(train_transforms, test_transforms)

    # train dataloader
    train_loader = torch.utils.data.DataLoader(trainset, **dataloader_args)

    # test dataloader
    test_loader = torch.utils.data.DataLoader(testset, **dataloader_args)

    return train_loader, test_loader
