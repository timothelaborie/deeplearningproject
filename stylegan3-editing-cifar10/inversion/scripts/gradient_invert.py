import pickle
import sys
sys.path.append(".")
sys.path.append("..")

import torch

import numpy as np
import os

from models.stylegan3.model import SG3Generator
from criteria.lpips.lpips import LPIPS
from PIL import Image
import torch.nn.functional as F

import torchvision.transforms as transforms
from criteria import id_loss, w_norm, moco_loss

import time

def tensor2im(var: torch.tensor):
    var = var.cpu().detach().transpose(0, 2).transpose(0, 1).numpy()
    var = ((var + 1) / 2)
    var[var < 0] = 0
    var[var > 1] = 1
    var = var * 255
    return Image.fromarray(var.astype('uint8'))

def main():
    decoder_path = '/cluster/home/bgunders/stylegan3-editing/pretrained_models/sg2c10-32.pkl'
    dir_path = '/cluster/home/bgunders/cifar10/train/'
    latents_path = '/cluster/home/bgunders/latents2.npy'
    out_path = '/cluster/scratch/bgunders/gradlatents/'

    #30 seconds per image
    start_index = 0
    end_index = 50000

    print("start", start_index)
    print("end", end_index)

    fnames = os.listdir(dir_path)
    fnames.sort()

    device = 'cuda'
    lpips_loss = LPIPS(net_type='vgg').to(device).eval()
    decoder = SG3Generator(checkpoint_path=decoder_path).decoder.cuda()
    
    latents = np.load(latents_path, allow_pickle=True).item()

    transform = transforms.Compose([
                    transforms.Resize((32, 32)),
                    transforms.ToTensor(),
                    transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])])

    all_latents = {}

    for fname in fnames[start_index:end_index]:
        print(fname)

        w_opt = torch.tensor(latents[fname][-1], dtype=torch.float32, device=device, requires_grad=True)

        optimizer = torch.optim.AdamW([w_opt], betas=(0.95, 0.999), lr=3e-2)
        w_opt = w_opt.unsqueeze(0)

        y = transform(Image.open(dir_path + fname).convert('RGB')).unsqueeze(0).cuda()
        start = time.time()

        min_loss = 9999999
        min_w = None

        for i in range(1000):
            y_hat = decoder.synthesis(w_opt, noise_mode='const', force_fp32=True)

            optimizer.zero_grad(set_to_none=True)
            loss = 0
            loss_lpips = lpips_loss(y_hat, y)

            loss += 0.9 * loss_lpips
            loss_l2 = F.mse_loss(y_hat, y)
            loss += 0.1 * loss_l2

            loss.backward()
            optimizer.step()
            if loss.item() < min_loss:
                min_loss = loss.item()
                min_w = w_opt.clone().detach()

        print(min_loss)
        print(time.time() - start)

        all_latents[fname] = min_w.squeeze(0)

    print("done")
    print(all_latents)

    np.save(out_path + f'grad_latents_{start_index:05}_{end_index:05}.npy', all_latents)

if __name__ == '__main__':
	main()
