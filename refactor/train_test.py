import numpy as np
from torch import optim
from torch.autograd import Variable
from torch.nn import MultiLabelSoftMarginLoss
import torch
import torch.nn as nn
from torch.optim.lr_scheduler import StepLR
from deepfool import deepfool
from utils import progress_bar


def mixup_data(x, y, device, alpha=1.0):
    lam = np.random.beta(alpha, alpha) if alpha > 0 else 1
    batch_size = x.size()[0]
    index = torch.randperm(batch_size).to(device)
    mixed_x = lam * x + (1 - lam) * x[index, :]
    y_a, y_b = y, y[index]
    return mixed_x, y_a, y_b, lam


def mixup_criterion(criterion, pred, y_a, y_b, lam):
    return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)


def train(model, device, train_loader, optimizer, specificity=""):
    model.train()
    train_loss = 0
    correct = 0
    total = 0
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        if specificity == "mixup":
            # Source : https://github.com/facebookresearch/mixup-cifar10
            inputs, targets_a, targets_b, lam = mixup_data(data, target, device=device, alpha=1.0)
            inputs, targets_a, targets_b = map(Variable, (inputs, targets_a, targets_b))
            outputs = model(inputs)
            loss = mixup_criterion(MultiLabelSoftMarginLoss(), outputs, targets_a, targets_b, lam)
            loss.backward()
            optimizer.step()
            train_loss += loss.data[0]
            _, predicted = torch.max(outputs.data, 1)
            total += target.size(0)
            correct += (lam * predicted.eq(targets_a.data).cpu().sum().float() + (1 - lam) * predicted.eq(targets_b.data).cpu().sum().float())
        else:
            output = model(data)
            loss = MultiLabelSoftMarginLoss()(output, target)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
            predictions = output.argmax(dim=1, keepdim=True)
            correct += predictions.eq(target.argmax(dim=1, keepdim=True).view_as(predictions)).sum().item()
            total += target.size(0)
        progress_bar(batch_idx, len(train_loader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)' % (train_loss / total, 100. * correct / total, correct, total))


def evaluate(model, device, data_loader, verbose=True):
    model.eval()
    loss = 0
    correct = 0
    total = 0
    with torch.no_grad():
        for batch_idx, (data, target) in enumerate(data_loader):
            data, target = data.to(device), target.to(device)
            output = model(data)
            loss += MultiLabelSoftMarginLoss()(output, target).item()
            predictions = output.argmax(dim=1, keepdim=True)
            target = target.argmax(dim=1, keepdim=True)
            correct += predictions.eq(target.view_as(predictions)).sum().item()
            total += target.size(0)
            if verbose:
                progress_bar(batch_idx, len(data_loader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)' % (loss / total, 100. * correct / total, correct, total))
    return correct / total, loss / total


def full_training(model, train_loader, val_loader, hyperparameters, device, specificity=""):
    optimizer = optim.Adam(model.parameters(), lr=hyperparameters["learning_rate"])
    scheduler = StepLR(optimizer, step_size=1, gamma=0.7)
    for epoch in range(hyperparameters["epochs"]):
        print("Epoch {}/{}".format(epoch, hyperparameters["epochs"]))
        print("Training ...")
        train(model, device, train_loader, optimizer, specificity=specificity)
        print("Evaluation on the validation set ...")
        evaluate(model, device, val_loader)
        scheduler.step()
        print("\n")


def score_report(model, device, val_loader, test_loader, blurred_test_loader):
    report = [['type', 'accuracy', 'loss', 'deep_fool']]
    print("Final report :")
    val_accuracy, val_loss = evaluate(model, device, val_loader, verbose=False)
    val_df_score = deepfool_score(model, device, val_loader)
    report.append(["val", val_accuracy, val_loss, val_df_score])
    print("\tPerformance on the validation set - acc. : {}, loss : {}, DeepFool score : {}".format(val_accuracy, val_loss, val_df_score))
    test_accuracy, test_loss = evaluate(model, device, test_loader, verbose=False)
    test_df_score = deepfool_score(model, device, test_loader)
    report.append(["test", test_accuracy, test_loss, test_df_score])
    print("\tPerformance on the testing set - acc. : {}, loss : {}, DeepFool score : {}".format(test_accuracy, test_loss, test_df_score))
    blurred_test_accuracy, blurred_test_loss = evaluate(model, device, blurred_test_loader, verbose=False)
    blurred_test_df_score = deepfool_score(model, device, blurred_test_loader)
    report.append(["blurred_test", blurred_test_accuracy, blurred_test_loss, blurred_test_df_score])
    print("\tPerformance on the blurred testing set - acc. : {}, loss : {}, DeepFool score : {}".format(blurred_test_accuracy, blurred_test_loss, blurred_test_df_score))
    return report


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
            norms.append(norm)
        batches_done += 1
        if batches_done % 50 == 0:
            break
    return np.mean(norms)
