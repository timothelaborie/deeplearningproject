

from __future__ import print_function
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import Dataset, DataLoader
from torch.autograd import Variable
from torchvision.utils import save_image

import matplotlib.pyplot as plt
import matplotlib.image as mpimg

import numpy as np

import tqdm


sets = []
sets.append(("train","mnist",datasets.MNIST('./data', train=True, download=True)))
sets.append(("test","mnist",datasets.MNIST('./data', train=False, download=True)))

sets.append(("train","cifar10",datasets.CIFAR10('./data', train=True, download=True)))
sets.append(("test","cifar10",datasets.CIFAR10('./data', train=False, download=True)))

sets.append(("train","fashionmnist",datasets.FashionMNIST('./data', train=True, download=True)))
sets.append(("test","fashionmnist",datasets.FashionMNIST('./data', train=False, download=True)))

for (split, name, dataset) in sets:

    data = np.array(dataset.data)
    labels = np.array(dataset.targets)
    #convert labels to one-hot
    one_hot_labels = np.zeros((labels.size, labels.max()+1))
    one_hot_labels[np.arange(labels.size),labels] = 1

    print(name, split, data.shape,labels.shape, one_hot_labels.shape)

    np.save("./datasets/" + name + "/" + split + ".npy", data)
    np.save("./datasets/" + name + "/" + split + "_labels.npy", one_hot_labels)


