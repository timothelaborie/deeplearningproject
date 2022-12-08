import random

import torch
import torch.nn as nn
import torch.nn.functional as F

from utils import mixup_data


def get_standard_model(dataset_name, device):
    if dataset_name.endswith("mnist"):
        return MnistNet(device=device)
    else:  # dataset_name.startswith("cifar")
        return CifarNet(device=device, n_out=(10 if dataset_name.endswith("10") else 100))


def get_vae(dataset_name, h_dim1, h_dim2, z_dim):
    if dataset_name.endswith("mnist"):
        return VAE(x_dim=28*28, h_dim1=h_dim1, h_dim2=h_dim2, z_dim=z_dim)
    else:  # dataset_name.startswith("cifar")
        return VAE(x_dim=32*32*3, h_dim1=h_dim1, h_dim2=h_dim2, z_dim=z_dim)


class MnistNet(nn.Module):
    def __init__(self, device):
        super(MnistNet, self).__init__()
        self.conv1 = nn.Conv2d(1, 16, 3, 1)
        self.conv2 = nn.Conv2d(16, 32, 3, 1)
        self.dropout = nn.Dropout(0.5)
        self.fc1 = nn.Linear(9216 // 2, 128)
        self.fc2 = nn.Linear(128, 10)
        self.softmax = nn.Softmax(dim=1)

        self.device = device

    def forward(self, x, target=None, mixup_hidden=False, mixup_alpha=1.0, layer_mix=None):
        if mixup_hidden:
            if layer_mix is None:
                layer_mix = random.randint(0, 3)
            out = x

            if layer_mix == 0:
                out, y_a, y_b, lam = mixup_data(out, target, self.device, mixup_alpha)
            out = self.conv1(out)
            out = F.relu(out)

            if layer_mix == 1:
                out, y_a, y_b, lam = mixup_data(out, target, self.device, mixup_alpha)
            out = self.conv2(out)
            out = F.relu(out)
            out = F.max_pool2d(out, 2)
            out = torch.flatten(out, 1)

            if layer_mix == 2:
                out, y_a, y_b, lam = mixup_data(out, target, self.device, mixup_alpha)
            out = self.fc1(out)
            out = F.relu(out)
            out = self.dropout(out)

            if layer_mix == 3:
                out, y_a, y_b, lam = mixup_data(out, target, self.device, mixup_alpha)
            out = self.fc2(out)
            out = self.softmax(out)
            return out, y_a, y_b, lam
        else:
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


class CifarNet(nn.Module):
    def __init__(self, device, n_out):
        super(CifarNet, self).__init__()
        self.device = device
        self.conv_layer_1 = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )
        self.conv_layer_2 = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Dropout2d(p=0.05),
        )
        self.conv_layer_3 = nn.Sequential(
            nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )
        self.fc_layer_1 = nn.Sequential(
            nn.Dropout(p=0.1),
            nn.Linear(4096, 1024),
            nn.ReLU(inplace=True),
        )
        self.fc_layer_2 = nn.Sequential(
            nn.Linear(1024, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.1),
        )
        self.fc_layer_3 = nn.Sequential(
            nn.Linear(512, n_out)
        )

    def forward(self, x, target=None, mixup_hidden=False, mixup_alpha=1.0, layer_mix=None):
        if mixup_hidden:
            if layer_mix is None:
                layer_mix = random.randint(0, 5)
            out = x

            if layer_mix == 0:
                out, y_a, y_b, lam = mixup_data(out, target, self.device, mixup_alpha)
            out = self.conv_layer_1(out)

            if layer_mix == 1:
                out, y_a, y_b, lam = mixup_data(out, target, self.device, mixup_alpha)
            out = self.conv_layer_2(out)

            if layer_mix == 2:
                out, y_a, y_b, lam = mixup_data(out, target, self.device, mixup_alpha)
            out = self.conv_layer_3(out)
            out = out.view(out.size(0), -1)

            if layer_mix == 3:
                out, y_a, y_b, lam = mixup_data(out, target, self.device, mixup_alpha)
            out = self.fc_layer_1(out)

            if layer_mix == 4:
                out, y_a, y_b, lam = mixup_data(out, target, self.device, mixup_alpha)
            out = self.fc_layer_2(out)

            if layer_mix == 5:
                out, y_a, y_b, lam = mixup_data(out, target, self.device, mixup_alpha)
            out = self.fc_layer_3(out)
            return out, y_a, y_b, lam
        else:
            x = self.conv_layer_1(x)
            x = self.conv_layer_2(x)
            x = self.conv_layer_3(x)
            x = x.view(x.size(0), -1)
            x = self.fc_layer_1(x)
            x = self.fc_layer_2(x)
            x = self.fc_layer_3(x)
            return x


class VAE(nn.Module):
    def __init__(self, x_dim, h_dim1, h_dim2, z_dim):
        super(VAE, self).__init__()
        self.x_dim = x_dim
        self.z_dim = z_dim
        # encoder part
        self.fc1 = nn.Linear(x_dim, h_dim1)
        self.fc2 = nn.Linear(h_dim1, h_dim2)
        self.fc31 = nn.Linear(h_dim2, z_dim)
        self.fc32 = nn.Linear(h_dim2, z_dim)
        # decoder part
        self.fc4 = nn.Linear(z_dim, h_dim2)
        self.fc5 = nn.Linear(h_dim2, h_dim1)
        self.fc6 = nn.Linear(h_dim1, x_dim)

    def encoder(self, x):
        h = F.relu(self.fc1(x))
        h = F.relu(self.fc2(h))
        return self.fc31(h), self.fc32(h)  # mu, log_var

    @staticmethod
    def sampling(mu, log_var):
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        return eps.mul(std).add_(mu)  # return z sample

    def decoder(self, z):
        h = F.relu(self.fc4(z))
        h = F.relu(self.fc5(h))
        return torch.sigmoid(self.fc6(h))

    def forward(self, x):
        mu, log_var = self.encoder(x.view(-1, self.x_dim))
        z = self.sampling(mu, log_var)
        return self.decoder(z), mu, log_var


def vae_loss_function(recon_x, x, mu, log_var, x_dim):
    bce = F.binary_cross_entropy(recon_x, x.view(-1, x_dim), reduction='sum')
    kld = -0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp())
    return bce + kld
