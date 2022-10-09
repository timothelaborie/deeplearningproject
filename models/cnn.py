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
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
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



def test(model, device, test_loader):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            # sum up batch loss
            test_loss += MultiLabelSoftMarginLoss()(output, target).item()
            # get the index of the max log-probability
            pred = output.argmax(dim=1, keepdim=True)
            # target is one-hot encoded, so argmax to get the index
            target = target.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)

    print('Test set: Average loss: {:.4f}, Accuracy: {}/{} ({:.5f}%)'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))




device = torch.device("cuda")
torch.manual_seed(1)


x1,y1 = np.load("./datasets/mnist_orig.npy"),np.load("./datasets/mnist_orig_labels.npy")
x2,y2 = np.load("./datasets/mixup.npy"),np.load("./datasets/mixup_labels.npy")

datasets = (x1,y1),(x2,y2)
for (x,y) in datasets:
    print("next dataset")
    # x = x[:batch_size*100]
    # y = y[:batch_size*100]
    x = x/x.max()
    x = x.reshape(x.shape[0],1,28,28)
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)
    #convert to torch tensors
    x_train = torch.from_numpy(x_train).float()
    y_train = torch.from_numpy(y_train).float()
    x_test = torch.from_numpy(x_test).float()
    y_test = torch.from_numpy(y_test).float()
    train_loader = DataLoader(torch.utils.data.TensorDataset(x_train,y_train), batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(torch.utils.data.TensorDataset(x_test,y_test), batch_size=batch_size, shuffle=True)

    model = Net().to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)

    scheduler = StepLR(optimizer, step_size=1, gamma=0.7)
    for epoch in range(1, 2):
        train(model, device, train_loader, optimizer, epoch)
        test(model, device, test_loader)
        scheduler.step()

    # to test adversarial robustness, we need to remove the softmax layer
    model.softmax = nn.Identity()

    # test the model on adversarial examples
    norms = []
    batches_done = 0
    for batch_idx, (data, target) in enumerate(test_loader):
        data, target = data.to(device), target.to(device)
        for (image,target) in zip(data,target):
            target = target.unsqueeze(0)
            # generate adversarial examples
            data = deepfool(image, model, num_classes=10, overshoot=0.02, max_iter=50)
            minimal_perturbation = data[0]
            # calculate the norm of the perturbation
            norm = np.linalg.norm(minimal_perturbation)
            # print(norm)
            norms.append(norm)
        batches_done += 1
        if batches_done % 50 == 0:
            break

    print('Average norm of perturbation needed: {:.5f}'.format(np.mean(norms)))



