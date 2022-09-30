import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.autograd import Variable
from torchvision.utils import save_image

import numpy as np

import tqdm

cuda_available = torch.cuda.is_available()

device = torch.device('cuda' if cuda_available else 'cpu')

pin_memory = True if cuda_available else False
batch_size = 128
latent_size = 1024
lr = 0.0003


#print(np.mean(train_dataset.data.numpy())/255)
#print(np.std(train_dataset.data.numpy())/255)


transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=0.5, std=0.5)
    ])

train_dataset = datasets.MNIST(root='./data/mnist/', train=True, transform=transform, download=True)
test_dataset = datasets.MNIST(root='./data/mnist/', train=False, transform=transform, download=False)


train = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
test = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False)


class Generator(nn.Module):
    def __init__(self):
        super().__init__()
        
        self.seq = nn.Sequential(
            nn.ConvTranspose2d(1024, 512, kernel_size=3, stride=2, padding=0, bias=False),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(),
            nn.ConvTranspose2d(512, 256, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(),
            nn.ConvTranspose2d(256, 128, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(),
            nn.ConvTranspose2d(128, 64, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(),
            nn.ConvTranspose2d(64, 1, kernel_size=3, stride=2, padding=1, bias=True),
            nn.Tanh()
            )
        
        
    def forward(self, x):
        x = self.seq(x)
        x = x[:, :, 0:28, 0:28]
        return x
        
class Discriminator(nn.Module):
    def __init__(self):
        super().__init__()
        
        self.seq = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(),
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(),
            nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(),
            nn.Conv2d(256, 512, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(),
            nn.Conv2d(512, 1,  kernel_size=3, stride=2, padding=1, bias=True),
            nn.Sigmoid()
            )
        
        
    def forward(self, x):
        x = self.seq(x)
        return x
        
G = Generator().to(device)

D = Discriminator().to(device)

def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv2d') != -1:
        nn.init.kaiming_normal_(m.weight.data, nonlinearity='leaky_relu')
    elif classname.find('ConvTranspose2d') != -1:
        nn.init.kaiming_normal_(m.weight.data, nonlinearity='leaky_relu')
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)
        
G = G.apply(weights_init)
D = D.apply(weights_init)

loss = nn.BCELoss()

G_opt = optim.AdamW(G.parameters(), lr=lr, betas=(0.5, 0.999))
D_opt = optim.AdamW(D.parameters(), lr=lr, betas=(0.5, 0.999))

def train_epoch(G_opt, D_opt, G, D, train):
    
    real_label = torch.tensor([1.0], device=device)
    fake_label = torch.tensor([0.0], device=device)
    
    G.train()
    D.train()
    
    dl = []
    gl = []
    
    for i, (x, _) in enumerate(tqdm.tqdm(train,0)):
        
        x_real = x.to(device)
        
        D_opt.zero_grad(set_to_none=True)
        
        D_out_real = D(x_real).view(-1)
        
        y_real = real_label.repeat(D_out_real.shape[0],)
        y_fake = fake_label.repeat(D_out_real.shape[0],)
        
        latent = torch.randn(D_out_real.shape[0], latent_size, 1, 1, device=device)
        
        
        with torch.no_grad():
            x_fake = G(latent)
        
        D_out_fake = D(x_fake).view(-1)
        
        D_real_loss = loss(D_out_real, y_real)
        D_fake_loss = loss(D_out_fake, y_fake)
        
        D_loss = D_real_loss + D_fake_loss
        D_loss.backward()
        D_opt.step()
        
        G_opt.zero_grad(set_to_none=True)
        
        x_fake = G(latent)
        D_out = D(x_fake).view(-1)
        
        G_loss = loss(D_out, y_real)
        
        G_loss.backward()
        G_opt.step()
        
        dl.append(D_loss.item())
        gl.append(G_loss.item())
        
    return np.mean(dl), np.mean(gl)


num = 1000

for img, _ in train:
    save_image(img, './samples/train.png')
    break

for epoch in range(0, num):
    dl, gl = train_epoch(G_opt, D_opt, G, D, train)
    print(gl, " ", dl)
    
    with torch.no_grad():
        test_z = Variable(torch.randn(batch_size, 1024, 1, 1).to(device))
        generated = G(test_z)
        save_image(generated, './samples/sample_' + str(epoch) + '.png')
        
    
            
    

