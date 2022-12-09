import numpy as np
from torch import optim
from torch.autograd import Variable
import torch
import torch.nn as nn
from torch.optim.lr_scheduler import StepLR
from deepfool import deepfool
from utils import progress_bar, mixup_criterion, mixup_data, DATASET_IMAGE_CHN, DATASET_IMAGE_DIM
from model import vae_loss_function


def train(model, device, train_loader, dataset_name, optimizer, specificity="", mixup_alpha=1.0, vae_model=None):
    criterion = nn.CrossEntropyLoss()
    model.train()
    train_loss = 0
    correct = 0
    total = 0
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        if specificity == "":
            # Standard training
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
            predictions = output.argmax(dim=1, keepdim=True)
            correct += predictions.eq(target.view_as(predictions)).sum().item()
            total += target.size(0)
        elif specificity in ["mixup", "manifold_mixup"]:
            # Mixup on intermediate representation (initial mixup is a special case)
            layer_mix = 0 if specificity == "mixup" else None
            outputs, targets_a, targets_b, lam = model(data, target=target, mixup_hidden=True, mixup_alpha=mixup_alpha, layer_mix=layer_mix)
            loss = mixup_criterion(criterion, outputs, targets_a, targets_b, lam)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += target.size(0)
            correct += (lam * predicted.eq(targets_a.data).cpu().sum().float() + (1 - lam) * predicted.eq(targets_b.data).cpu().sum().float())
        elif specificity == "mixup_vae":
            # Mixup on the latent codes
            assert vae_model is not None, "No VAE model has been provided"
            inputs, targets_a, targets_b, lam = mixup_data(data, target, device=device, alpha=mixup_alpha)
            inputs, targets_a, targets_b = map(Variable, (inputs, targets_a, targets_b))
            inputs = vae_model.decoder(inputs).to(device).view(inputs.shape[0], DATASET_IMAGE_CHN[dataset_name], DATASET_IMAGE_DIM[dataset_name], DATASET_IMAGE_DIM[dataset_name])
            outputs = model(inputs)
            loss = mixup_criterion(criterion, outputs, targets_a, targets_b, lam)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += target.size(0)
            correct += (lam * predicted.eq(targets_a.data).cpu().sum() + (1 - lam) * predicted.eq(targets_b.data).cpu().sum())
        else:
            assert False, "Unknown specificity {}".format(specificity)
        progress_bar(batch_idx, len(train_loader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)' % (train_loss / total, 100. * correct / total, correct, total))


def evaluate(model, device, data_loader, verbose=True):
    criterion = nn.CrossEntropyLoss()
    model.eval()
    loss = 0
    correct = 0
    total = 0
    with torch.no_grad():
        for batch_idx, (data, target) in enumerate(data_loader):
            data, target = data.to(device), target.to(device)
            output = model(data)
            loss += criterion(output, target).item()
            predictions = output.argmax(dim=1, keepdim=True)
            correct += predictions.eq(target.view_as(predictions)).sum().item()
            total += target.size(0)
            if verbose:
                progress_bar(batch_idx, len(data_loader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)' % (loss / total, 100. * correct / total, correct, total))
    return correct / total, loss / total


def full_training(model, train_loader, val_loader, dataset_name, hyperparameters, device, specificity="", mixup_alpha=1.0, vae_model=None):
    optimizer = optim.Adam(model.parameters(), lr=hyperparameters["learning_rate"])
    scheduler = StepLR(optimizer, step_size=1, gamma=0.7)
    for epoch in range(hyperparameters["epochs"]):
        print("Epoch {}/{}".format(epoch, hyperparameters["epochs"]))
        print("Training ...")
        train(model, device, train_loader, dataset_name, optimizer, specificity=specificity, mixup_alpha=mixup_alpha, vae_model=vae_model)
        print("Evaluation on the validation set ...")
        evaluate(model, device, val_loader)
        scheduler.step()
        print("\n")


def vae_train(vae, device, optimizer, train_loader):
    vae.train()
    train_loss = 0
    total = 0
    for batch_idx, (data, target) in enumerate(train_loader):
        data = data.to(device)
        optimizer.zero_grad()
        recon_batch, mu, log_var = vae(data)
        loss = vae_loss_function(recon_batch, data, mu, log_var, vae.x_dim)
        loss.backward()
        train_loss += loss.item()
        total += target.size(0)
        optimizer.step()
        progress_bar(batch_idx, len(train_loader), 'Loss: %.3f' % (train_loss / total))


def vae_evaluate(vae, device, data_loader, verbose=True):
    vae.eval()
    test_loss = 0
    total = 0
    with torch.no_grad():
        for batch_idx, (data, target) in enumerate(data_loader):
            data = data.to(device)
            recon, mu, log_var = vae(data)
            total += target.size(0)
            test_loss += vae_loss_function(recon, data, mu, log_var, vae.x_dim).item()
            if verbose:
                progress_bar(batch_idx, len(data_loader), 'Loss: %.3f' % (test_loss / total))
    return test_loss / total


def full_vae_training(vae, train_loader, val_loader, device, hyperparameters):
    optimizer = optim.Adam(vae.parameters())
    for epoch in range(hyperparameters["vae_epochs"]):
        print("Epoch {}/{}".format(epoch, hyperparameters["vae_epochs"]))
        print("Training ...")
        vae_train(vae, device, optimizer, train_loader)
        print("Evaluation on the validation set ...")
        vae_evaluate(vae, device, val_loader, verbose=True)
        print("\n")


def score_report(model, device, val_loader, test_loader, blurred_test_loader):
    report = [['type', 'accuracy', 'loss', 'deep_fool']]
    print("Final report :")
    val_accuracy, val_loss = evaluate(model, device, val_loader, verbose=False)
    val_df_score = deepfool_score(model, device, val_loader)
    report.append(["val", val_accuracy, val_loss, val_df_score])
    print("\tPerformance on the validation set - acc. : {:0.4f}, loss : {:.4f}, DeepFool score : {:.4f}".format(val_accuracy, val_loss, val_df_score))
    test_accuracy, test_loss = evaluate(model, device, test_loader, verbose=False)
    test_df_score = deepfool_score(model, device, test_loader)
    report.append(["test", test_accuracy, test_loss, test_df_score])
    print("\tPerformance on the testing set - acc. : {:0.4f}, loss : {:.4f}, DeepFool score : {:.4f}".format(test_accuracy, test_loss, test_df_score))
    blurred_test_accuracy, blurred_test_loss = evaluate(model, device, blurred_test_loader, verbose=False)
    report.append(["blurred_test", blurred_test_accuracy, blurred_test_loss, float('nan')])
    print("\tPerformance on the blurred testing set - acc. : {:0.4f}, loss : {:.4f}".format(blurred_test_accuracy, blurred_test_loss))
    return report


def deepfool_score(model, device, test_loader):
    # return 0  # This is only to speed up testing
    model.softmax = nn.Identity()
    # test the model on adversarial examples
    norms = []
    batches_done = 0
    for batch_idx, (data, target) in enumerate(test_loader):
        data, target = data.to(device), target.to(device)
        for image, t in zip(data, target):
            t = t.unsqueeze(0)
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
