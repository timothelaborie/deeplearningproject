import torch
import torch.nn as nn

import numpy as np

import tqdm
from torch import optim
from torch.autograd import Variable
from torchvision.utils import save_image

import utils


class GAN:
    def __init__(self, args, device, generator, discriminator):
        self.args = args

        self.device = device

        self.generator = generator(self.args).to(device)
        self.discriminator = discriminator(self.args).to(device)

        print(self.generator)
        print(self.discriminator)

        if self.args.dp is None:
            self.generator.apply(utils.weights_init)
            self.discriminator.apply(utils.weights_init)
        else:
            self.generator.load_state_dict(torch.load(self.args.gp))
            self.discriminator.load_state_dict(torch.load(self.args.dp))

        self.generator_opt = optim.AdamW(self.generator.parameters(), lr=self.args.lr,
                                         betas=(self.args.beta1, self.args.beta2))
        self.discriminator_opt = optim.AdamW(self.discriminator.parameters(), lr=self.args.lr,
                                             betas=(self.args.beta1, self.args.beta2))

    def get_generator(self):
        return self.generator

    def get_discriminator(self):
        return self.discriminator

    def train_epoch(self, train):
        loss = torch.nn.BCELoss()

        device = self.device

        real_label = torch.tensor([1.0], device=device)
        fake_label = torch.tensor([0.0], device=device)

        self.generator.train()
        self.discriminator.train()

        dl = []
        gl = []

        for i, (x, _) in enumerate(tqdm.tqdm(train, 0)):
            x_real = x.to(device)

            self.generator_opt.zero_grad(set_to_none=True)

            D_out_real = self.discriminator(x_real).view(-1)

            y_real = real_label.repeat(D_out_real.shape[0], )
            y_fake = fake_label.repeat(D_out_real.shape[0], )

            latent = self.get_noise(D_out_real.size()[0])

            with torch.no_grad():
                x_fake = self.generator(latent)

            D_out_fake = self.discriminator(x_fake).view(-1)

            D_real_loss = loss(D_out_real, y_real)
            D_fake_loss = loss(D_out_fake, y_fake)

            D_loss = D_real_loss + D_fake_loss
            D_loss.backward()
            self.discriminator_opt.step()

            self.generator_opt.zero_grad(set_to_none=True)

            x_fake = self.generator(latent)
            D_out = self.discriminator(x_fake).view(-1)

            G_loss = loss(D_out, y_real)

            G_loss.backward()
            self.generator_opt.step()

            dl.append(D_loss.item())
            gl.append(G_loss.item())

        return np.mean(dl), np.mean(gl)

    def train(self, train_data):
        for epoch in range(self.args.epochs):
            dl, gl = self.train_epoch(train_data)
            print(f"Epoch: {epoch}, D_loss: {dl}, G_loss: {gl}")
            torch.save(self.generator.state_dict(), f"./gancheckpoints/generator_{epoch}.pth")
            torch.save(self.discriminator.state_dict(), f"./gancheckpoints/discriminator_{epoch}.pth")
            with torch.no_grad():
                test_z = Variable(torch.randn(self.args.batch_size, 1024, 1, 1).to(self.device))
                generated = self.generator(test_z)
                save_image(generated, './samples/sample_' + str(epoch) + '.png')

    def get_noise(self, batch_size):
        torch.randn(batch_size.shape[0], self.args.latent_size, 1, 1, device=self.args.device)
        assert False