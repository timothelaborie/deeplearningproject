
import torch
import torch.nn as nn
from models import gan


class Generator(nn.Module):
    def __init__(self, args):
        super().__init__()

        self.args = args

        ls = self.args.latent_size
        
        self.seq = nn.Sequential(
            nn.ConvTranspose2d(ls, int(ls/2), kernel_size=3, stride=2, padding=0, bias=False),
            nn.BatchNorm2d(int(ls/2)),
            nn.LeakyReLU(),
            nn.ConvTranspose2d(int(ls/2), int(ls/4), kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(int(ls/4)),
            nn.LeakyReLU(),
            nn.ConvTranspose2d(int(ls/4), int(ls/8), kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(int(ls/8)),
            nn.LeakyReLU(),
            nn.ConvTranspose2d(int(ls/8), int(ls/16), kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(int(ls/16)),
            nn.LeakyReLU(),
            nn.ConvTranspose2d(int(ls/16), 1, kernel_size=3, stride=2, padding=1, bias=True),
            nn.Tanh()
            )

    def forward(self, x):
        x = self.seq(x)
        x = x[:, :, 0:28, 0:28]
        return x


class Discriminator(nn.Module):
    def __init__(self, args):
        super().__init__()

        self.args = args

        ls = self.args.latent_size
        
        self.seq = nn.Sequential(
            nn.Conv2d(1, int(ls/16), kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(int(ls/16)),
            nn.LeakyReLU(),
            nn.Conv2d(int(ls/16), int(ls/8), kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(int(ls/8)),
            nn.LeakyReLU(),
            nn.Conv2d(int(ls/8), int(ls/4), kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(int(ls/4)),
            nn.LeakyReLU(),
            nn.Conv2d(int(ls/4), int(ls/2), kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(int(ls/2)),
            nn.LeakyReLU(),
            nn.Conv2d(int(ls/2), 1,  kernel_size=3, stride=2, padding=1, bias=True),
            nn.Sigmoid()
            )

    def forward(self, x):
        x = self.seq(x)
        return x


class GAN5Conv(gan.GAN):
    def __init__(self, args, device):
        super().__init__(args, device, Generator, Discriminator)

    def get_noise(self, batch_size):
        return torch.randn(batch_size, self.args.latent_size, 1, 1, device=self.device)



