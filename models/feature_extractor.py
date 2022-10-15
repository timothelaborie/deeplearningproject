from __future__ import print_function
import argparse
import torch
import torch.nn as nn
from torch.nn import MultiLabelSoftMarginLoss
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import DataLoader
import numpy as np
from deepfool import deepfool


batch_size=16
lr = 0.001

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 16, 3, 1)
        self.conv2 = nn.Conv2d(16, 32, 3, 1)
        self.dropout = nn.Dropout(0.5)
        self.fc1 = nn.Linear(9216//2, 128)
        self.fc2 = nn.Linear(128, 10)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(x)
        x = self.conv2(x)
        x = F.relu(x)
        x = F.max_pool2d(x, 2)
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.dropout(x)
        x = self.fc2(x)
        output = self.softmax(x)
        return output


def train(model, device, train_loader, optimizer, epoch):
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        #use MultiLabelSoftMarginLoss
        loss = MultiLabelSoftMarginLoss()(output, target)
        loss.backward()
        optimizer.step()
        if batch_idx % 1000 == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.item()))


device = torch.device("cuda")
torch.manual_seed(1)

load = lambda x: np.load("../datasets/mnist/" + x + ".npy")

x_train = load("mnist")
y_train = load("mnist_labels")
x_train = x_train/x_train.max()
x_train = x_train.reshape(x_train.shape[0],1,28,28)
x_train = torch.from_numpy(x_train).float()
y_train = torch.from_numpy(y_train).float()

train_loader = DataLoader(torch.utils.data.TensorDataset(x_train,y_train), batch_size=batch_size, shuffle=True)

model = Net().to(device)
optimizer = optim.Adam(model.parameters(), lr=lr)

scheduler = StepLR(optimizer, step_size=1, gamma=0.7)
for epoch in range(1, 2):
    train(model, device, train_loader, optimizer, epoch)
    scheduler.step()



# this model will be used as a feature extractor
model.softmax = nn.Identity()
model.fc2 = nn.Identity()


model = model.cpu()

#save the model
torch.save(model, "./feature_extractor.pt")