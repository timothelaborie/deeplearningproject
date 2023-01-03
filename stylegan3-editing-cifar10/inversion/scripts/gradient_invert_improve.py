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

import imageio

import random

import matplotlib.pyplot as plt

def main():
    decoder_path = '/cluster/home/bgunders/dl_inversion_data/sg2c10-32.pkl'
    dir_path = '/cluster/scratch/bgunders/cifar_10_small_test/train'
    latents_path = 'latents2.npy'
    out_path = ''

    start_index = 0
    end_index = 1

    print("start", start_index)
    print("end", end_index)

    fnames = os.listdir(dir_path)
    fnames.sort()

    device = 'cuda'
    lpips_loss = LPIPS(net_type='vgg').to(device).eval()
    decoder = SG3Generator(checkpoint_path=decoder_path).decoder.cuda()
    
    #latents = np.load(latents_path, allow_pickle=True).item()

    transform = transforms.Compose([
                    transforms.Resize((32, 32)),
                    transforms.ToTensor(),
                    transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])])

    all_latents = {}
    latents = []

    for fname in fnames[start_index:end_index]:
        print(fname)

        w_opt = torch.tensor(latents[fname][-1], dtype=torch.float32, device=device, requires_grad=True)
        

        optimizer = torch.optim.AdamW([w_opt], betas=(0.95, 0.999), lr=3e-2)
        w_opt = w_opt.repeat((8, 1))
        print(w_opt)
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

            #print("loss", loss)
            #print("lpips", loss_lpips.item())

        
        print(min_loss)
        print(time.time() - start)

        y_hat = decoder.synthesis(min_w, noise_mode='const', force_fp32=True)
        im = tensor2im(y_hat[0])
        im.save("ims2/" + fname)

        all_latents[fname] = min_w.squeeze(0).cpu().numpy()

    print("done")
    print(all_latents)

    np.save(out_path + f'grad_latents_{start_index:05}_{end_index:05}.npy', all_latents)

def generate_mp4(out_name, images, kwargs):
    writer = imageio.get_writer(str(out_name) + '.mp4', **kwargs)
    for image in images:
        writer.append_data(np.array(image))
    writer.close()

def combine_gl():
    gl_path = '/home/ben/play/gradlatents/'

    fnames = os.listdir(gl_path)
    fnames.sort()

    print(fnames)

    all_latents = {}

    for fname in fnames:
        cl = np.load(gl_path + fname, allow_pickle=True).item()

        for key in cl.keys():
            print(key)
            all_latents[key] = cl[key]

    np.save(f'grad_latents_{00000}_{50000}.npy', all_latents)

def tensor2im(var: torch.tensor):
    var = var.cpu().detach().transpose(0, 2).transpose(0, 1).numpy()
    var = ((var + 1) / 2)
    var[var < 0] = 0
    var[var > 1] = 1
    var = var * 255
    return Image.fromarray(var.astype('uint8'))

if __name__ == '__main__':
    """ device = 'cuda'
    decoder_path = 'pretrained_models/sg2c10-32.pkl'
    dir_path = '/home/ben/cifar10/train/'
    latents_path = 'grad_latents_00000_50000.npy'
    out_path = '/cluster/scratch/bgunders/' """


    device = 'cuda'
    decoder_path = '/cluster/home/bgunders/dl_inversion_data/sg2c10-32.pkl'
    dir_path = '/cluster/scratch/bgunders/cifar_10_small_test/train/'
    latents_path = '/cluster/home/bgunders/dl_inversion_data/grad_latents_00000_50000_0.025.npy'
    out_path = '/cluster/scratch/bgunders/'

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
        #print(fname, flush=True)

        w_init = torch.tensor(l[fname], dtype=torch.float32, device=device, requires_grad=False).unsqueeze(0).cuda()

        y1 = transform(Image.open(dir_path + fname).convert('RGB')).unsqueeze(0).cuda()
        y2 = decoder.synthesis(w_init, noise_mode='const', force_fp32=True)

        m = ((y1 - y2) ** 2).mean()
        #print(F.mse_loss(y2, y1))
        means.append(m.cpu())

        if m > 0.02:
            
            print(m)
            count += 1
            print(fname, flush=True)
            
            w_opt = torch.tensor(l[fname], dtype=torch.float32, device=device, requires_grad=True)
            
            optimizer = torch.optim.AdamW([w_opt], betas=(0.95, 0.999), lr=3e-2)
            w_opt = w_opt.unsqueeze(0)
            
            y = transform(Image.open(dir_path + fname).convert('RGB')).unsqueeze(0).cuda()
            start = time.time()

            min_loss = 9999999
            min_w = None

            for i in range(4000):
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

                #print("loss", loss)
                #print("lpips", loss_lpips.item())
                #print("mse", F.mse_loss(y, y_hat))

            print("inversion")
            print(min_loss)
            print(time.time() - start, flush=True)

            l[fname] = min_w.squeeze(0).cpu().numpy()

    means.sort()
    print(count)
    np.save(out_path + f'grad_latents_00000_50000_0.02.npy', l)

