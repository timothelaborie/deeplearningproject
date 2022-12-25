import pickle
import torch


                    
import torch
import torch.nn as nn
import torchvision

import torchvision.transforms as transforms

from torchvision.utils import save_image

import os


import subprocess

transform = transforms.Compose([transforms.ToTensor()])

trainset = torchvision.datasets.CIFAR10(root='./data', train=True,
                                        download=True, transform=transform)

trainloader = torch.utils.data.DataLoader(trainset, batch_size=1,
                                            shuffle=False, num_workers=2)

testset = torchvision.datasets.CIFAR10(root='./data', train=False,
                                        download=True, transform=transform)

testloader = torch.utils.data.DataLoader(testset, batch_size=1,
                                            shuffle=False, num_workers=2)


def save_cifar10():
    trainpath = "/home/ben/cifar10/train/"
    testpath = "/home/ben/cifar10/test/"
    
    for i, (x, y) in enumerate(trainloader):
        save_image(x, f'{trainpath}{i:05}_{int(y)}.png')

    for i, (x, y) in enumerate(testloader):
        save_image(x, f'{testpath}{i:05}_{int(y)}.png')


def main():

    #save_cifar10()

    with open('stylegan2-cifar10-32x32.pkl', 'rb') as f:
        G = pickle.load(f)['G']

    print(G)

    torch.save(G.state_dict(), 'stylegan2-cifar10-32x32.pt') 


if __name__ == "__main__":
    main()

    