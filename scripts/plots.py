import pickle
import sys
sys.path.append(".")
sys.path.append("..")

import torch

import numpy as np
import os

from sg3 import SG3Generator
from PIL import Image
import torch.nn.functional as F

import torchvision.transforms as transforms

import time

import imageio

import random

import matplotlib.pyplot as plt

def tensor2im(var: torch.tensor):
    var = var.cpu().detach().transpose(0, 2).transpose(0, 1).numpy()
    var = ((var + 1) / 2)
    var[var < 0] = 0
    var[var > 1] = 1
    var = var * 255
    return Image.fromarray(var.astype('uint8'))

if __name__ == '__main__':
    device = 'cuda'
    decoder_path = 'pretrained_models/sg2c10-32.pkl'
    dir_path = '/home/ben/cifar10/train/'
    latents_path = 'grad_latents_00000_50000_0.025.npy'
    out_path = '/cluster/scratch/bgunders/'


    """ device = 'cuda'
    decoder_path = '/cluster/home/bgunders/dl_inversion_data/sg2c10-32.pkl'
    dir_path = '/cluster/scratch/bgunders/cifar_10_small_test/train/'
    latents_path = '/cluster/home/bgunders/dl_inversion_data/grad_latents_00000_50000.npy'
    out_path = '/cluster/scratch/bgunders/' """

    decoder = SG3Generator(checkpoint_path=decoder_path).decoder.eval().cuda()
    l = np.load(latents_path, allow_pickle=True).item()


    """ for k in list(l.keys()):
        l0 = torch.tensor(l[k]).unsqueeze(0).cuda()
        y_hat = decoder.synthesis(l0 , noise_mode='const', force_fp32=True)
        im = tensor2im(y_hat[0])
        im.save("outims/" + k) """
	

    fnames = os.listdir(dir_path)
    fnames.sort()

    transform = transforms.Compose([
                    transforms.Resize((32, 32)),
                    transforms.ToTensor(),
                    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
                    ])

    means = []

    count = 0

    lpips_loss = LPIPS(net_type='vgg').to(device).eval()

    import matplotlib.pyplot as plt

    for fname in fnames:
        print(fname, flush=True)

        w_init = torch.tensor(l[fname], dtype=torch.float32, device=device, requires_grad=False).unsqueeze(0).cuda()

        y1 = transform(Image.open(dir_path + fname).convert('RGB')).unsqueeze(0).cuda()
        y2 = transform(tensor2im(decoder.synthesis(w_init, noise_mode='const', force_fp32=True)[0])).unsqueeze(0).cuda()

        m = (((y1 - y2) ** 2)).mean()
        print(m)
        means.append(m.cpu())

    means.sort()
    plt.plot(means)
    plt.title("mse, original and inversion, normalized, optimize 0.02")
    plt.xlabel("sorted by mse")
    plt.ylabel("mse")
    plt.show()
