from torchvision import datasets as ds
from torch.utils.data import DataLoader
from torchvision import transforms as T
import torchvision as tv
import torch
import torch.nn as nn
import math
import numpy as np
from torch.autograd import Variable
from torch import optim
import torch.backends.cudnn as cudnn

import vgg16bn

import tqdm


from torchvision.transforms.functional import InterpolationMode

import device_decision
device = device_decision.device


train_transform = T.Compose(
    [
        T.AutoAugment(policy=T.AutoAugmentPolicy.CIFAR10, interpolation=InterpolationMode.BILINEAR,),
        T.RandomHorizontalFlip(),
        T.ToTensor(),
        T.Normalize(
            mean=[0.49139968, 0.48215827, 0.44653124],
            std=[0.24703233, 0.24348505, 0.26158768]),
        T.RandomCrop(size=(28, 28))
    ]
)



test_transform = T.Compose(
    [
        T.ToTensor(),
        T.Normalize(
            mean=[0.49139968, 0.48215827, 0.44653124],
            std=[0.24703233, 0.24348505, 0.26158768]),
        T.CenterCrop(size=(28, 28)),
    ]
)

def get_loaders():
    t = train_transform
    train_set = ds.CIFAR10(root='./data',
                    train=True, transform=t, download=True)


    test_set = ds.CIFAR10(root='./data',
                                train=False, download=True, transform=test_transform)

    pin_memory = True if device == 'cuda:0' else False
    num_workers = 4

    train_loader = torch.utils.data.DataLoader(train_set, batch_size=64, shuffle=True,pin_memory=pin_memory, num_workers=num_workers)
    test_loader = torch.utils.data.DataLoader(test_set, batch_size=64, shuffle=False, pin_memory=pin_memory, num_workers=num_workers)

    return train_loader, test_loader

def get_model_classifier(PATH):
    model = vgg.VGG().to(device)
    model.load_state_dict(torch.load(PATH,map_location=device))
    model.fc = nn.Sequential()

    return model
