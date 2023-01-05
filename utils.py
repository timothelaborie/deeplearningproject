import os
import sys
import time
import numpy as np
import csv
import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import platform

from torchvision.utils import save_image

DATASET_IMAGE_DIM = {"mnist":  28, "fashionmnist": 28, "cifar10": 32, "cifar100": 32}
DATASET_IMAGE_CHN = {"mnist":  1, "fashionmnist": 1, "cifar10": 3, "cifar100": 3}
DATASETS_CALLS = {"mnist": datasets.MNIST, "fashionmnist": datasets.FashionMNIST, "cifar10": datasets.CIFAR10, "cifar100": datasets.CIFAR100}
DATASETS_TRAIN_SPLITS = {"mnist": [50000, 10000], "fashionmnist": [50000, 10000], "cifar10": [40000, 10000], "cifar100": [40000, 10000]}


def store_png_images(image_tensors, file_name, dataset_name):
    data = image_tensors / 2 + 0.5 if dataset_name.startswith("cifar") else image_tensors
    save_image(data.view(data.shape[0], DATASET_IMAGE_CHN[dataset_name], DATASET_IMAGE_DIM[dataset_name], DATASET_IMAGE_DIM[dataset_name]), file_name)


def get_dataset(dataset_name, hyperparameters=None, blur=False):
    transformations = [transforms.ToTensor()]
    if dataset_name.startswith("cifar"):
        transformations.append(transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)))
        if hyperparameters is not None:
            if hyperparameters["augment"] == "cifar10":
                train_transforms = transforms.Compose(
                    [
                        transforms.AutoAugment(policy=transforms.AutoAugmentPolicy.CIFAR10),
                        transforms.RandomHorizontalFlip(),
                        transforms.ToTensor(),
                        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                        transforms.RandomCrop(size=(28, 28))
                    ]
                )

                test_transforms = transforms.Compose(
                    [
                        transforms.ToTensor(),
                        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                        transforms.CenterCrop(size=(28, 28))
                    ]
                )

                train = DATASETS_CALLS[dataset_name]('./data', train=True, download=True, transform=test_transforms)
                _, val = torch.utils.data.random_split(train, DATASETS_TRAIN_SPLITS[dataset_name], generator=torch.Generator().manual_seed(42))
                train = DATASETS_CALLS[dataset_name]('./data', train=True, download=True, transform=train_transforms)
                train, _ = torch.utils.data.random_split(train, DATASETS_TRAIN_SPLITS[dataset_name], generator=torch.Generator().manual_seed(42))
                test = DATASETS_CALLS[dataset_name]('./data', train=False, download=True, transform=test_transforms)
                return train, val, test
    if dataset_name.startswith("mnist"):
        transformations.append(transforms.Normalize((0.5,), (0.5,)))
    if blur:
        transformations.append(transforms.GaussianBlur(5, 5))
    train = DATASETS_CALLS[dataset_name]('./data', train=True, download=True, transform=transforms.Compose(transformations))
    test = DATASETS_CALLS[dataset_name]('./data', train=False, download=True, transform=transforms.Compose(transformations))
    train, val = torch.utils.data.random_split(train, DATASETS_TRAIN_SPLITS[dataset_name], generator=torch.Generator().manual_seed(42))
    return train, val, test


def mixup_data(x, y, device, alpha=1.0,lam=-1):
    if lam == -1:
        lam = np.random.beta(alpha, alpha) if alpha > 0 else 1
    batch_size = x.size()[0]
    index = torch.randperm(batch_size).to(device)
    mixed_x = lam * x + (1 - lam) * x[index, :]
    y_a, y_b = y, y[index]
    return mixed_x, y_a, y_b, lam


def mixup_criterion(criterion, prediction, y_a, y_b, lam):
    return lam * criterion(prediction, y_a) + (1 - lam) * criterion(prediction, y_b)


def show_cuda_info():
    print("torch.cuda.is_available() : {}".format(torch.cuda.is_available()))
    print("torch.cuda.device_count() : {}".format(torch.cuda.device_count()))
    print("torch.cuda.current_device() : {}".format(torch.cuda.current_device()))
    print("torch.cuda.get_device_name(0) : {}".format(torch.cuda.get_device_name(0)))


def dir_exists(dir_name):  # Returns whether the given name is an existing directory
    return os.path.isdir(dir_name)


def file_exists(file_name):  # Returns whether the given name is an existing file
    return os.path.isfile(file_name)


def create_directory(directory_name):  # Creates a new directory given the name if possible
    try:
        os.mkdir(directory_name)
    except FileExistsError:
        pass


def write_csv(file_name, data):
    with open(file_name, 'w', encoding='UTF8') as f:
        writer = csv.writer(f)
        for row in data:
            writer.writerow(row)


def read_csv(file_name):
    with open(file_name, 'r') as f:
        csv_reader = csv.reader(f)
        for row in csv_reader:
            print(row)


_, term_width = 0, 80  # os.popen('stty size', 'r').read().split()
term_width = int(term_width)
TOTAL_BAR_LENGTH = 40
last_time = time.time()
begin_time = last_time


# Source : https://github.com/facebookresearch/mixup-cifar10
def progress_bar(current, total, msg=None):
    global last_time, begin_time
    if current == 0:
        begin_time = time.time()  # Reset for new bar.
    cur_len = int(TOTAL_BAR_LENGTH*current/total)
    rest_len = int(TOTAL_BAR_LENGTH - cur_len) - 1
    sys.stdout.write(' [')
    for i in range(cur_len):
        sys.stdout.write('=')
    sys.stdout.write('>')
    for i in range(rest_len):
        sys.stdout.write('.')
    sys.stdout.write(']')
    cur_time = time.time()
    step_time = cur_time - last_time
    last_time = cur_time
    tot_time = cur_time - begin_time
    L = []
    L.append('  Step: %s' % format_time(step_time))
    L.append(' | Tot: %s' % format_time(tot_time))
    if msg:
        L.append(' | ' + msg)
    msg = ''.join(L)
    sys.stdout.write(msg)
    for i in range(term_width-int(TOTAL_BAR_LENGTH)-len(msg)-3):
        sys.stdout.write(' ')
    if current < total-1:
        sys.stdout.write('\r')
    else:
        sys.stdout.write('\n')
    sys.stdout.flush()


# Source : https://github.com/facebookresearch/mixup-cifar10
def format_time(seconds):
    days = int(seconds / 3600/24)
    seconds = seconds - days*3600*24
    hours = int(seconds / 3600)
    seconds = seconds - hours*3600
    minutes = int(seconds / 60)
    seconds = seconds - minutes*60
    secondsf = int(seconds)
    seconds = seconds - secondsf
    millis = int(seconds*1000)
    f = ''
    i = 1
    if days > 0:
        f += str(days) + 'D'
        i += 1
    if hours > 0 and i <= 2:
        f += str(hours) + 'h'
        i += 1
    if minutes > 0 and i <= 2:
        f += str(minutes) + 'm'
        i += 1
    if secondsf > 0 and i <= 2:
        f += str(secondsf) + 's'
        i += 1
    if millis > 0 and i <= 2:
        f += str(millis) + 'ms'
        i += 1
    if f == '':
        f = '0ms'
    return f


def mask_dict(initial_dict, keys):
    return {k: initial_dict[k] for k in initial_dict if k in keys}


def hyperparameter_to_name(hyperparameters):
    return "&".join(sorted(["{}꞉{}".format(k, hyperparameters[k]) for k in hyperparameters]))


def report_results():
    print("Showing the results stored on disk")
    for dataset in ["mnist", "fashionmnist", "cifar10", "cifar100"]:
        for variant in ["standard", "mixup", "manifold_mixup", "vae", "gan"]:
            dir_name = "./results/{}/{}".format(dataset, variant)
            if dir_exists(dir_name):
                dir_list = os.listdir(dir_name)
                if len(dir_list) > 0:
                    print("\n---Results for dataset {} and variant {}---".format(dataset, variant))
                runs_aggregations = dict()
                for result_file in dir_list:
                    seed_index = result_file.find("random_seed")
                    partial_file_name = result_file[:seed_index-1] + result_file[seed_index+13:]
                    seed = int(result_file[seed_index+12])
                    with open("{}/{}".format(dir_name, result_file), newline='') as csvfile:
                        report = np.zeros((3, 3))
                        line = 0
                        for row in csv.reader(csvfile):
                            if line > 0:
                                report[line-1, :] = np.array([float(f) for f in row[1:]])
                            line += 1
                    if partial_file_name not in runs_aggregations:
                        runs_aggregations[partial_file_name] = []
                    runs_aggregations[partial_file_name].append((seed, report))
                full_reports = []
                for run in runs_aggregations:
                    seeds, reports = zip(*runs_aggregations[run])
                    reports = np.array(reports)
                    seeds = [str(i) for i in sorted(list(seeds))]
                    full_reports.append((run, seeds, reports.mean(axis=0), reports.std(axis=0)))
                full_reports = sorted(full_reports, key=lambda r: -r[2][0][0])  # Sort with respect to the mean validation accuracy
                for index, report in enumerate(full_reports):
                    print("Rank {}".format(index))
                    print("\tParameters : {}".format(report[0][:-4].replace("&", ", ").replace(":", ": ")))
                    print("\tSeeds : {} (n: {})".format(", ".join(report[1]), len(report[1])))
                    for i, split in enumerate(["Val", "Test", "Blurred test"]):
                        for j, metric in enumerate(["accuracy", "loss", "DeepFool score"]):
                            print("\t{} {} : {:.3f} (+/- {:.3f})".format(split, metric, report[2][i, j], report[3][i, j]))

