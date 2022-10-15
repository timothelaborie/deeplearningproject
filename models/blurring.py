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
import cv2

import matplotlib.pyplot as plt
import matplotlib.image as mpimg

import numpy as np



x = np.load("./datasets/mnist/test.npy")

# blur the images using gaussian blur
for i in range(x.shape[0]):
    blurred = cv2.GaussianBlur(x[i], (5,5), 0)
    # fig, ax = plt.subplots(1,2)
    # ax[0].imshow(x[i])
    # ax[1].imshow(blurred)
    x[i] = blurred
    if i % 1000 == 9:
        print(i)
        # break

#save the blurred images
np.save("./datasets/mnist/test_blurred.npy", x)