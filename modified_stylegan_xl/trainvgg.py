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

import utils



def train_one_epoch(model, criterion, optimizer, data_loader, device, epoch):
    model.train()

    for i, (image, target) in enumerate(tqdm.tqdm(data_loader, 0)):
        optimizer.zero_grad()
        image, target = image.to(device), target.to(device)
        output = model(image)

        loss = criterion(output, target)

        loss.backward()
        optimizer.step()

def evaluate(model, criterion, data_loader, device, epoch):
    model.eval()

    val_loss = 0

    total_agree = 0
    total = 0
    correct = 0

    with torch.inference_mode():
        for image, target in data_loader:
            image = image.to(device, non_blocking=True)
            target = target.to(device, non_blocking=True)
            pred = model(image)
            
            loss = criterion(pred, target)

            total += image.shape[0]

            _, predicted = torch.max(pred, 1)
            correct += (target == predicted).sum().item()
            val_loss = val_loss + loss * target.shape[0]

    print(f'Epoch {epoch} \t\t Validation Loss: {val_loss / len(data_loader)} \t\t val_acc: {correct / total}')
    return correct / total

def get_vgg():
    model = vgg16bn.VGG()
    model = model.to(device)
    return model

def train_vgg():
    model = get_vgg()
    print(model)


    train, test = utils.get_loaders()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.1))

    criterion = nn.CrossEntropyLoss().to(device)
    val_acc = evaluate(model, criterion, test, device, -1)
    for epoch in range(0, 270):
        train_one_epoch(model, criterion, optimizer, train, device, epoch)
        val_acc = evaluate(model, criterion, test, device, epoch)
        lr_scheduler.step()
        torch.save(model.state_dict(), "vgg16.pth")

if __name__ == "__main__":
    train_vgg()