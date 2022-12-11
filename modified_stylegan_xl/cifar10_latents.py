import torch
import torch.nn as nn
import torchvision

import torchvision.transforms as transforms

from torchvision.utils import save_image

import os

import click

import subprocess

transform = transforms.Compose([transforms.ToTensor()])

trainset = torchvision.datasets.CIFAR10(root='/cluster/home/bgunders/stylegan_xl/data', train=True,
                                        download=False, transform=transform)

trainloader = torch.utils.data.DataLoader(trainset, batch_size=1,
                                            shuffle=False, num_workers=2)


def save_cifar10(savedir):
    
    os.makedirs(outdir, exist_ok=True)
    for i, (x, y) in enumerate(trainloader):
        save_image(x, f'{savedir}{i:05}_{int(y)}.png')


@click.command()
@click.option('--save-cifar', 'save_cifar', help='save_cifar', type=bool, default=False)
@click.option('--root', 'root', help='root', type=str, default='/cluster/home/bgunders/stylegan_xl/')
@click.option('--savedir', 'savedir', help='savedir cifar', type=str, default='cifar10/')
@click.option('--start', 'start', help='start index', type=int, default=-1)
@click.option('--end', 'end', help='end index', type=int, default=-1)
def main(save_cifar, root, savedir, start, end):
    if save_cifar:
        save_cifar10(savedir)

    assert start < end

    for i, (x, y) in enumerate(trainloader):
        if i < start:
            continue
        if i >= end:
            break
        subprocess.run([f'python', f'{root}cifar10_run_inversion.py', f'--outdir=/cluster/scratch/bgunders/inv/', '--target', f'{root}{savedir}{i:05}_{int(y)}.png', '--inv-steps','0','--run-pti','--pti-steps','200','--save-video','false' ,'--network=https://s3.eu-central-1.amazonaws.com/avg-projects/stylegan_xl/models/cifar10.pkl'])


if __name__ == "__main__":
    main()

    
