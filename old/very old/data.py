from torchvision import datasets, transforms
import torch
from torch.utils.data import random_split


def get_default_transform(mean=0.5, std=0.5):
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std)
    ])
    return transform


def get_data(dataset='mnist', batch_size=128, transform=get_default_transform(), val_perc=0.3):
    if dataset == 'mnist':
        train_dataset = datasets.MNIST(root='./data/mnist/', train=True, transform=transform, download=True)
        test_dataset = datasets.MNIST(root='./data/mnist/', train=False, transform=transform, download=False)
    else:
        assert False

    train_len = len(train_dataset)
    val_len = int(train_len * val_perc)
    train_len -= val_len

    train_dataset, val_dataset = random_split(train_dataset, [train_len, val_len],
                                              generator=torch.Generator().manual_seed(42))

    train = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
    val = torch.utils.data.DataLoader(dataset=val_dataset, batch_size=batch_size, shuffle=False)
    test = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False)

    return train, val, test
