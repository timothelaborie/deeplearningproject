import os
import sys
import time

import cv2
from torch.utils.data import DataLoader
import torch
from torchvision import datasets
import numpy as np
import csv


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


DATASETS_CALLS = {"mnist": datasets.MNIST, "fashionmnist": datasets.FashionMNIST, "cifar10": datasets.CIFAR10, "cifar100": datasets.CIFAR100}
DATASETS_TRAIN_SPLITS = {"mnist": [50000, 10000], "fashionmnist": [50000, 10000], "cifar10": [40000, 10000], "cifar100": [40000, 10000]}


def download_dataset(dataset_name):
    assert dataset_name in DATASETS_CALLS.keys(), "Dataset {} is not available".format(dataset_name)
    create_directory("./datasets")
    if dir_exists("./datasets/{}".format(dataset_name)):
        print("Dataset {} is already stored on disk".format(dataset_name))
    else:
        print("Downloading dataset {}".format(dataset_name))
        dataset_call = DATASETS_CALLS[dataset_name]
        dir_name = "./datasets/{}".format(dataset_name)
        create_directory(dir_name)
        for train, split_name in [(True, "train"), (False, "test")]:
            dataset = dataset_call('../data', train=train, download=True)
            data = np.array(dataset.data)
            labels = np.array(dataset.targets)
            print("\tSplit : {}".format(split_name))
            print("\t\tData shape : {}".format(data.shape))
            print("\t\tLabel shape : {}".format(labels.shape))
            print("\t\tNumber of labels : {}".format(labels.max() + 1))
            one_hot_labels = np.zeros((labels.size, labels.max() + 1))
            one_hot_labels[np.arange(labels.size), labels] = 1  # convert labels to one-hot
            np.save("{}/{}_features.npy".format(dir_name, split_name), data)
            np.save("{}/{}_labels.npy".format(dir_name, split_name), one_hot_labels)


def load_dataset(name, train_val, specificity=""):
    split = "train" if train_val else "test"
    if specificity == "":  # Standard dataset
        x = np.load("./datasets/{}/{}_features.npy".format(name, split))
        y = np.load("./datasets/{}/{}_labels.npy".format(name, split))
    elif specificity in ["blurred"]:
        x = np.load("./augmented_datasets/{}/{}_{}_features.npy".format(name, specificity, split))
        y = np.load("./augmented_datasets/{}/{}_{}_labels.npy".format(name, specificity, split))
    else:
        assert False, "Specificity {} is unknown".format(specificity)
    x = x / x.max()
    if name.endswith("mnist"):
        x = x.reshape(x.shape[0], 1, 28, 28)
    else:  # name.startswith("cifar")
        x = x.reshape(x.shape[0], 3, 32, 32)
    dataset = torch.utils.data.TensorDataset(torch.from_numpy(x).float(), torch.from_numpy(y).float())
    if train_val:
        return torch.utils.data.random_split(dataset, DATASETS_TRAIN_SPLITS[name])  # train_set, val_set
    else:
        return dataset


def blur_images(dataset_name):
    assert dataset_name in DATASETS_CALLS.keys(), "Dataset {} is not available".format(dataset_name)
    create_directory("./augmented_datasets")
    dir_name = "./augmented_datasets/{}".format(dataset_name)
    if dir_exists(dir_name):
        print("Blurred {} dataset is already stored on disk".format(dataset_name))
    else:
        print("Blurring dataset {}".format(dataset_name))
        create_directory(dir_name)
        for split_name in ["train", "test"]:
            x = np.load("./datasets/{}/{}_features.npy".format(dataset_name, split_name))
            y = np.load("./datasets/{}/{}_labels.npy".format(dataset_name, split_name))
            # blur the images using gaussian blur
            for i in range(x.shape[0]):
                blurred = cv2.GaussianBlur(x[i], (5, 5), 0)
                # fig, ax = plt.subplots(1,2)
                # ax[0].imshow(x[i])
                # ax[1].imshow(blurred)
                x[i] = blurred
            # save the blurred images
            np.save("{}/blurred_{}_features.npy".format(dir_name, split_name), x)
            np.save("{}/blurred_{}_labels.npy".format(dir_name, split_name), y)


_, term_width = 0, 180  # os.popen('stty size', 'r').read().split()
term_width = int(term_width)
TOTAL_BAR_LENGTH = 86.
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
    # Go back to the center of the bar.
    for i in range(term_width-int(TOTAL_BAR_LENGTH/2)):
        sys.stdout.write('\b')
    sys.stdout.write(' %d/%d ' % (current+1, total))
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
