import os
import cv2
from torch.utils.data import DataLoader
import torch
from torchvision import datasets
import numpy as np


def show_cuda_info():
    print("torch.cuda.is_available() : {}".format(torch.cuda.is_available()))
    print("torch.cuda.device_count() : {}".format(torch.cuda.device_count()))
    print("torch.cuda.current_device() : {}".format(torch.cuda.current_device()))
    print("torch.cuda.get_device_name(0) : {}".format(torch.cuda.get_device_name(0)))


def dir_exists(dir_name):  # Returns whether the given name is an existing directory
    return os.path.isdir(dir_name)


def create_directory(directory_name):  # Creates a new directory given the name if possible
    try:
        os.mkdir(directory_name)
    except FileExistsError:
        pass


def download_datasets(verbose=False):
    create_directory("./datasets")
    for dataset_name, dataset_call in [("mnist", datasets.MNIST), ("fashionmnist", datasets.FashionMNIST), ("cifar10", datasets.CIFAR10), ("cifar100", datasets.CIFAR100)]:
        dir_name = "./datasets/{}".format(dataset_name)
        create_directory(dir_name)
        for train, split_name in [(True, "train"), (False, "test")]:
            dataset = dataset_call('../data', train=train, download=True)
            data = np.array(dataset.data)
            labels = np.array(dataset.targets)
            if verbose:
                print("Dataset : {}, split : {}".format(dataset_name, split_name))
                print("\tData shape : {}".format(data.shape))
                print("\tLabel shape : {}".format(labels.shape))
                print("\tNumber of labels : {}".format(labels.max() + 1))
            # convert labels to one-hot
            one_hot_labels = np.zeros((labels.size, labels.max() + 1))
            one_hot_labels[np.arange(labels.size), labels] = 1
            np.save("{}/{}_features.npy".format(dir_name, split_name), data)
            np.save("{}/{}_labels.npy".format(dir_name, split_name), one_hot_labels)


def load_dataset(name, train, specificity=""):
    split = "train" if train else "test"
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
    return torch.utils.data.TensorDataset(torch.from_numpy(x).float(), torch.from_numpy(y).float())


def blur_images(dataset_name):
    create_directory("./augmented_datasets")
    dir_name = "./augmented_datasets/{}".format(dataset_name)
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
