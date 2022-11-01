# prerequisites
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.autograd import Variable
from torchvision.utils import save_image
import numpy as np
from os.path import exists
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

batch_size = 100
latent_size = 20

load = lambda x: np.load("./datasets/mnist/" + x + ".npy")
X = load("train")
y = load("train_labels")
X = X/X.max()
X = X.reshape(X.shape[0],1,28,28)

dataset = torch.utils.data.TensorDataset(torch.from_numpy(X).float(), torch.from_numpy(y).float())

train_loader = torch.utils.data.DataLoader(dataset=dataset, batch_size=batch_size, shuffle=True)


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



#todo change the code below to work with the gan












# build model
vae = VAE(x_dim=784, h_dim1= 512, h_dim2=256, z_dim=latent_size)
if torch.cuda.is_available():
    vae.cuda()

optimizer = optim.Adam(vae.parameters())
# return reconstruction error + KL divergence losses
def loss_function(recon_x, x, mu, log_var):
    BCE = F.binary_cross_entropy(recon_x, x.view(-1, 784), reduction='sum')
    KLD = -0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp())
    return BCE + KLD

def train(epoch):
    vae.train()
    train_loss = 0
    for batch_idx, (data, _) in enumerate(train_loader):
        data = data.cuda()
        optimizer.zero_grad()
        
        recon_batch, mu, log_var = vae(data)
        loss = loss_function(recon_batch, data, mu, log_var)
        
        loss.backward()
        train_loss += loss.item()
        optimizer.step()
        
        if batch_idx % 100 == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.item() / len(data)))
    print('====> Epoch: {} Average loss: {:.4f}'.format(epoch, train_loss / len(train_loader.dataset)))

def test():
    vae.eval()
    test_loss= 0
    with torch.no_grad():
        for data, _ in test_loader:
            data = data.cuda()
            recon, mu, log_var = vae(data)
            
            # sum up batch loss
            test_loss += loss_function(recon, data, mu, log_var).item()
        
    test_loss /= len(test_loader.dataset)
    print('====> Test set loss: {:.4f}'.format(test_loss))

# retrain = True
retrain = False

if retrain:
    for epoch in range(1, 51):
        train(epoch)
        test()
    # save the model
    torch.save(vae.state_dict(), './models/vae.pth')

else:
    vae.load_state_dict(torch.load('./models/vae.pth'))



#save a sample
with torch.no_grad():
    z = torch.randn(64, latent_size).cuda()
    sample = vae.decoder(z).cuda()
    save_image(sample.view(64, 1, 28, 28), './models/sample_' + '.png')

# generate batches of images with their corresponding latent vectors as the labels
# if not exists("./datasets/mnist/vae_images.npy"):
X = []
y = []
with torch.no_grad():
    for i in range(500):
        z = torch.randn(batch_size, latent_size).cuda()
        sample = vae.decoder(z).cuda()
        image = sample.view(batch_size, 1, 28, 28).cpu().numpy()
        X.append(image)
        y.append(z.cpu().numpy())

    X = np.concatenate(X, axis=0)
    y = np.concatenate(y, axis=0)

print(X.shape, y.shape)
X = np.array(X, dtype=np.uint8)
np.save('./datasets/mnist/vae_images.npy', X)
np.save('./datasets/mnist/vae_latent.npy', y)


#train a model to find the nearest latent vector to a given image
batch_size=16
lr = 0.001

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 16, 3, 1)
        self.conv2 = nn.Conv2d(16, 32, 3, 1)
        self.dropout = nn.Dropout(0.5)
        self.fc1 = nn.Linear(9216//2, 128)
        self.fc2 = nn.Linear(128, latent_size)

    def forward(self, x):
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
        return x


def train(model, device, train_loader, optimizer, epoch):
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = F.mse_loss(output, target)
        loss.backward()
        optimizer.step()
        if batch_idx % 1000 == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.item()))


#load data
X = np.load('./datasets/mnist/vae_images.npy')
y = np.load('./datasets/mnist/vae_latent.npy')

X = X/X.max()

X = torch.from_numpy(X).float()
y = torch.from_numpy(y).float()

train_loader = DataLoader(torch.utils.data.TensorDataset(X,y), batch_size=batch_size, shuffle=True)
model = Net().cuda()

#train the model
# retrain = True
retrain = False

if retrain:
    optimizer = optim.Adam(model.parameters(), lr=lr)

    for epoch in range(1, 51):
        train(model, 'cuda', train_loader, optimizer, epoch)

    #save the model
    torch.save(model.state_dict(), './models/vae_finder.pth')

else:
    model.load_state_dict(torch.load('./models/vae_finder.pth'))







# (experimental) this section checks if using a feature extractor improves the latent vector
mse = []

#generate a random image and find the nearest latent vector
feature_extractor = torch.load('./models/feature_extractor_mnist.pt').cuda()
for i in range(10):
    z = torch.randn(1, latent_size).cuda()
    original = vae.decoder(z).cuda().view(1, 1, 28, 28)


    #find the nearest latent vector
    z = model(original).cuda()

    #clean the tensors
    z = torch.from_numpy(z.cpu().detach().numpy()).float().cuda()
    original = torch.from_numpy(original.cpu().detach().numpy()).float().cuda()

    #the estimated latent vector can be improved using gradient descent
    optimizer = optim.Adam([z], lr=0.01)
    for i in range(5):
        optimizer.zero_grad()
        gen = vae.decoder(z).cuda().view(1, 1, 28, 28)
        loss = F.mse_loss(feature_extractor(original), feature_extractor(gen))
        loss.backward()
        optimizer.step()

    #plot the original image and the reconstructed image
    with torch.no_grad():
        reconstructed = vae.decoder(z).cuda()
        # fig, ax = plt.subplots(1,2)
        # ax[0].imshow(original.view(28,28).cpu().numpy())
        # ax[1].imshow(reconstructed.view(28,28).cpu().numpy())

        # print(feature_extractor(reconstructed.view(1, 1, 28, 28)).cpu().numpy())
        mse.append(F.mse_loss(original, reconstructed.view(1, 1, 28, 28)).item())
        
print("mean:",np.mean(mse))





#generate new training data
print("generating new training data")
x,y = np.load("./datasets/mnist/train.npy"), np.load("./datasets/mnist/train_labels.npy")
print(x.shape)
print(y.shape)
generated_images = []
generated_labels = []
for i in range(0,100):
    #select 2 images at random
    idx1 = np.random.randint(0,x.shape[0])
    image1 = x[idx1]
    label1 = y[idx1]
    idx2 = np.random.randint(0,x.shape[0])
    image2 = x[idx2]
    label2 = y[idx2]

    #find the nearest latent vector to each image
    image1 = torch.from_numpy(image1).float().view(1,1,28,28).cuda()
    image2 = torch.from_numpy(image2).float().view(1,1,28,28).cuda()
    z1 = model(image1).cuda()
    z2 = model(image2).cuda()

    #clean the tensors
    z1 = torch.from_numpy(z1.cpu().detach().numpy()).float().cuda()
    z2 = torch.from_numpy(z2.cpu().detach().numpy()).float().cuda()
    image1 = torch.from_numpy(image1.cpu().detach().numpy()).float().cuda()
    image2 = torch.from_numpy(image2.cpu().detach().numpy()).float().cuda()

    #the estimated latent vector can be improved using gradient descent
    optimizer1 = optim.Adam([z1], lr=0.01)
    optimizer2 = optim.Adam([z2], lr=0.01)
    for j in range(5):
        optimizer1.zero_grad()
        optimizer2.zero_grad()
        gen1 = vae.decoder(z1).cuda().view(1, 1, 28, 28)
        gen2 = vae.decoder(z2).cuda().view(1, 1, 28, 28)
        loss1 = F.mse_loss(feature_extractor(image1), feature_extractor(gen1))
        loss2 = F.mse_loss(feature_extractor(image2), feature_extractor(gen2))
        loss = loss1 + loss2
        loss.backward()
        optimizer1.step()
        optimizer2.step()
        

    #generate a new image between the two
    with torch.no_grad():
        alpha = 0.9
        z = alpha*z1 + (1-alpha)*z2
        new_image = vae.decoder(z).cuda()
        new_image = new_image.view(28,28).cpu().numpy()
        #interpolate the labels
        new_label = alpha*label1 + (1-alpha)*label2

        generated_images.append(new_image)
        generated_labels.append(new_label)

    if i%100==0:
        print(i)

generated_images = np.array(generated_images)
generated_labels = np.array(generated_labels)
x = np.concatenate([x,generated_images], axis=0)
y = np.concatenate([y,generated_labels], axis=0)
print(x.shape, y.shape)
# print(y[60000:60100])

#save the new training data
x = np.array(x, dtype=np.uint8)
np.save("./datasets/mnist/vae.npy", x)
np.save("./datasets/mnist/vae_labels.npy", y)