import cv2
from torch.utils.data import DataLoader
import torch
from torchvision import datasets
import numpy as np
from utils import create_directory, dir_exists

DATASET_IMAGE_DIM = {"mnist":  28, "fashionmnist": 28, "cifar10": 32, "cifar100": 32}
DATASET_IMAGE_CHN = {"mnist":  1, "fashionmnist": 1, "cifar10": 3, "cifar100": 3}
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
