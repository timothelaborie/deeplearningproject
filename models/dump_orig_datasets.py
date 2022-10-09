

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


dataset = datasets.MNIST('../data', train=True, download=True)
data = dataset.data
labels = dataset.targets
#convert labels to one-hot
one_hot_labels = torch.zeros(len(labels), 10)
one_hot_labels[torch.arange(len(labels)), labels] = 1

print(data.shape)
print(one_hot_labels.shape)

np.save("./datasets/mnist_orig.npy", data.numpy())
np.save("./datasets/mnist_orig_labels.npy", one_hot_labels.numpy())
