
import torch
import torch.nn as nn

def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv2d') != -1:
        nn.init.kaiming_normal_(m.weight.data, nonlinearity='leaky_relu')
    elif classname.find('ConvTranspose2d') != -1:
        nn.init.kaiming_normal_(m.weight.data, nonlinearity='leaky_relu')
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)

def get_opt(G, D, lr=0.0003, beta1=0.5, beta2=0.999):
    G_opt = optim.AdamW(G.parameters(), lr=lr, betas=(beta1, beta2))
    D_opt = optim.AdamW(D.parameters(), lr=lr, betas=(beta1, beta2))

    return G_opt, D_opt