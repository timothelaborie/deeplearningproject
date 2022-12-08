
import argparse

import data
import models.dcgan

import torch

model_dict = {
    'dcgan5': models.dcgan.GAN5Conv
}

parser = argparse.ArgumentParser(description='latent mixup')
# lines taken from https://github.com/yaohungt/Barlow-Twins-HSIC/blob/main/linear.py
parser.add_argument('--vae', action='store_true', help='use vae instread of gan')
parser.add_argument('--dp', type=str, default=None, help='discrimintaor checkpoint')
parser.add_argument('--gp', type=str, default=None, help='generator checkpoint')
parser.add_argument('--batch_size', default=128, type=int, help='batch size')
parser.add_argument('--latent_size', default=1024, type=int, help='batch size')
parser.add_argument('--model', type=str, choices=model_dict.keys(), default='dcgan5', help='choose model')
parser.add_argument('--data', type=str, choices=['mnist'], default='mnist', help='choose which dataset shoudl be used')
parser.add_argument('--epochs', default=50, type=int, help='number of epochs to train for')
parser.add_argument('--beta1', default=0.5, type=float, help='beta1 of adam(w)')
parser.add_argument('--beta2', default=0.999, type=float, help='beta2 of adam(w)')
parser.add_argument('--lr', default=0.0003, type=float, help='learning rate of adam(w)')
parser.add_argument('--weight_decay', default=0.0, type=float, help='adam weight decay')

cuda_available = torch.cuda.is_available()

device = torch.device('cuda' if cuda_available else 'cpu')


def main():
    args = parser.parse_args()
    print(args)

    model = model_dict[args.model](args, device)

    train, val, test = data.get_data()

    if args.dp is None:
        model.train(train)

if __name__ == '__main__':
    main()