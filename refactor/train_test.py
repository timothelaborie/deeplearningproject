import numpy as np
from torch import optim
from torch.autograd import Variable
from torch.nn import MultiLabelSoftMarginLoss
import torch
import torch.nn as nn
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import DataLoader
from deepfool import deepfool


def mixup_data(x, y, device, alpha=1.0):
    lam = np.random.beta(alpha, alpha) if alpha > 0 else 1
    batch_size = x.size()[0]
    index = torch.randperm(batch_size).to(device)
    mixed_x = lam * x + (1 - lam) * x[index, :]
    y_a, y_b = y, y[index]
    return mixed_x, y_a, y_b, lam


def mixup_criterion(criterion, pred, y_a, y_b, lam):
    return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)


def train(model, device, train_loader, optimizer, epoch, specificity=""):
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        if specificity == "mixup":
            inputs, targets_a, targets_b, lam = mixup_data(data, target, device=device, alpha=1.0)
            inputs, targets_a, targets_b = map(Variable, (inputs, targets_a, targets_b))
            outputs = model(inputs)
            loss = mixup_criterion(MultiLabelSoftMarginLoss(), outputs, targets_a, targets_b, lam)
            loss.backward()
            optimizer.step()
            # train_loss += loss.data[0]
            # _, predicted = torch.max(outputs.data, 1)
            # total += targets.size(0)
            # correct += (lam * predicted.eq(targets_a.data).cpu().sum().float() + (1 - lam) * predicted.eq(targets_b.data).cpu().sum().float())
        else:
            output = model(data)
            loss = MultiLabelSoftMarginLoss()(output, target)
            loss.backward()
            optimizer.step()
        # if batch_idx % 1000 == 0:
        #     print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(epoch, batch_idx * len(data), len(train_loader.dataset), 100. * batch_idx / len(train_loader), loss.item() / train_loader.batch_size))


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
            predictions = output.argmax(dim=1, keepdim=True)
            # target is one-hot encoded, so argmax to get the index
            target = target.argmax(dim=1, keepdim=True)
            correct += predictions.eq(target.view_as(predictions)).sum().item()
    test_loss /= len(test_loader.dataset)
    print('Test set: Average loss: {:.4f}, Accuracy: {}/{} ({:.5f}%)'.format(test_loss, correct, len(test_loader.dataset), 100. * correct / len(test_loader.dataset)))


def full_training(model, train_dataset, test_dataset, epochs, batch_size, learning_rate, device, specificity=""):
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = StepLR(optimizer, step_size=1, gamma=0.7)
    for epoch in range(epochs):
        print("Epoch {}/{}".format(epoch, epochs))
        train(model, device, train_loader, optimizer, epoch, specificity=specificity)
        test(model, device, train_loader)
        scheduler.step()
    deepfool_score(model, device, test_loader)


def deepfool_score(model, device, test_loader):
    model.softmax = nn.Identity()
    # test the model on adversarial examples
    norms = []
    batches_done = 0
    for batch_idx, (data, target) in enumerate(test_loader):
        data, target = data.to(device), target.to(device)
        for image, target in zip(data, target):
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
    print('DeepFool - Average norm of perturbation needed: {:.5f}'.format(np.mean(norms)))
    return np.mean(norms)
