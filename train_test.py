import random
import numpy as np
from torch import optim
from torch.autograd import Variable
import torch
import torch.nn as nn
from torch.optim.lr_scheduler import StepLR,ReduceLROnPlateau
import torchvision.transforms.functional as F
import torch.nn.functional as F2
from deepfool import deepfool
from utils import progress_bar, mixup_criterion, mixup_data, DATASET_IMAGE_CHN, DATASET_IMAGE_DIM
from model import vae_loss_function,GAN
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from torchvision.utils import save_image
import tqdm

from torchvision import datasets, transforms
from PIL import Image
def gentr_fn(alist):
    while 1:
        for j in alist:
            yield j


def train(model, device, image_train_loader, dataset_name, optimizer, hyperparameters, specificity="", mixup_alpha=1.0, mixup_ratio=1.0, vae_model=None,gan_model=None, latent_train_loader=None):
    criterion = nn.CrossEntropyLoss()
    model.train()
    train_loss = 0
    correct = 0
    total = 0
    loader_size = len(image_train_loader)
    # assert latent_train_loader is None or loader_size == len(latent_train_loader), "Loaders don't have the same size"
    image_it = iter(image_train_loader)
    latent_it = None if latent_train_loader is None else gentr_fn(latent_train_loader)
    for batch_idx in range(loader_size):
        optimizer.zero_grad()
        image_data, image_target = next(image_it)
        latent_data, latent_target = (None, None) if latent_it is None else next(latent_it)
        data, target = image_data, image_target
        data, target = data.to(device), target.to(device)
        if specificity == "" or random.random() > mixup_ratio:
            # Standard training
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
            predictions = output.argmax(dim=1, keepdim=True)
            correct += predictions.eq(target.view_as(predictions)).sum().item()
            total += target.size(0)
        elif specificity in ["mixup", "manifold_mixup"]:
            # Mixup on intermediate representation (initial mixup is a special case)
            layer_mix = 0 if specificity == "mixup" else None
            outputs, targets_a, targets_b, lam = model(data, target=target, mixup_hidden=True, mixup_alpha=mixup_alpha, layer_mix=layer_mix)
            loss = mixup_criterion(criterion, outputs, targets_a, targets_b, lam)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += target.size(0)
            correct += (lam * predicted.eq(targets_a.data).cpu().sum().float() + (1 - lam) * predicted.eq(targets_b.data).cpu().sum().float())
        elif specificity == "mixup_vae":
            data, target = latent_data, latent_target
            data, target = data.to(device), target.to(device)
            # Mixup on the latent codes
            assert vae_model is not None, "No VAE model has been provided"
            inputs, targets_a, targets_b, lam = mixup_data(data, target, device=device, alpha=mixup_alpha)
            inputs, targets_a, targets_b = map(Variable, (inputs, targets_a, targets_b))
            inputs = vae_model.decoder(inputs).to(device).view(inputs.shape[0], DATASET_IMAGE_CHN[dataset_name], DATASET_IMAGE_DIM[dataset_name], DATASET_IMAGE_DIM[dataset_name])
            if hyperparameters["vae_sharp"] > 1:
                inputs = F.adjust_sharpness(inputs, hyperparameters["vae_sharp"])  # Sharpen the output of the VAE
            outputs = model(inputs)
            loss = mixup_criterion(criterion, outputs, targets_a, targets_b, lam)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += target.size(0)
            correct += (lam * predicted.eq(targets_a.data).cpu().sum() + (1 - lam) * predicted.eq(targets_b.data).cpu().sum())
        elif specificity == "mixup_gan":
            data, target = latent_data, latent_target
            data, target = data.to(device), target.to(device)
            # Mixup on the latent codes
            assert gan_model is not None, "No GAN model has been provided"
            inputs, targets_a, targets_b, lam = mixup_data(data, target, device=device, alpha=mixup_alpha)
            inputs, targets_a, targets_b = map(Variable, (inputs, targets_a, targets_b))

            train_transforms = transforms.Compose(
                [
                    transforms.AutoAugment(policy=transforms.AutoAugmentPolicy.CIFAR10),
                    transforms.RandomHorizontalFlip(),
                    transforms.ToTensor(),
                    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                    transforms.RandomCrop(size=(28, 28))
                ]
            )

            if dataset_name == "cifar10":
                with torch.no_grad():
                    inputs = gan_model.synthesis(inputs, noise_mode='const', force_fp32=True).to(device).view(inputs.shape[0], DATASET_IMAGE_CHN[dataset_name], DATASET_IMAGE_DIM[dataset_name], DATASET_IMAGE_DIM[dataset_name])

                    if hyperparameters["augment"] == "cifar10":
                        inputs = ((inputs + 1) / 2).cpu().transpose(1, 3).transpose(1, 2).numpy()
                        inputs[inputs < 0] = 0
                        inputs[inputs > 1] = 1
                        inputs = inputs * 255

                        new_inputs = torch.empty((inputs.shape[0], 3, 28, 28), dtype=torch.float32)

                        for i, inp in enumerate(inputs):
                            inp = train_transforms(Image.fromarray(inp.astype('uint8')))

                            new_inputs[i, :, :, :] = inp
                            

                        inputs = new_inputs.cuda()
            else:
                with torch.no_grad():
                    inputs = gan_model.generator(inputs)

            outputs = model(inputs)
            loss = mixup_criterion(criterion, outputs, targets_a, targets_b, lam)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += target.size(0)
            correct += (lam * predicted.eq(targets_a.data).cpu().sum() + (1 - lam) * predicted.eq(targets_b.data).cpu().sum())
        else:
            assert False, "Unknown specificity {}".format(specificity)
        # progress_bar(batch_idx, loader_size, 'Loss: %.3f | Acc: %.3f%% (%d/%d)' % (train_loss / total, 100. * correct / total, correct, total))


def evaluate(model, device, data_loader, verbose=False):
    criterion = nn.CrossEntropyLoss()
    model.eval()
    loss = 0
    correct = 0
    total = 0
    with torch.no_grad():
        for batch_idx, (data, target) in enumerate(data_loader):
            data, target = data.to(device), target.to(device)
            output = model(data)
            loss += criterion(output, target).item()
            predictions = output.argmax(dim=1, keepdim=True)
            correct += predictions.eq(target.view_as(predictions)).sum().item()
            total += target.size(0)
            if verbose:
                progress_bar(batch_idx, len(data_loader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)' % (loss / total, 100. * correct / total, correct, total))
    return correct / total, loss / total


def full_training(model, image_train_loader, val_loader, dataset_name, hyperparameters, device, specificity="", mixup_alpha=1.0, mixup_ratio=1.0, vae_model=None,gan_model=None, latent_train_loader=None):
    if hyperparameters["optim"] == "sgd":
        print("using SGD")
        optimizer = optim.SGD(model.parameters(), lr=hyperparameters["learning_rate"], momentum=hyperparameters["momentum"], weight_decay=hyperparameters["weight_decay"])
        scheduler = ReduceLROnPlateau(optimizer, 'min', factor=hyperparameters["gamma"], patience=10)
    else:
        optimizer = optim.Adam(model.parameters(), lr=hyperparameters["learning_rate"])
        scheduler = StepLR(optimizer, step_size=1, gamma=hyperparameters["gamma"])
    for epoch in range(hyperparameters["epochs"]):
        print("Epoch {}/{}".format(epoch, hyperparameters["epochs"]))
        print("Training ...")
        train(model, device, image_train_loader, dataset_name, optimizer, hyperparameters, specificity=specificity, mixup_alpha=mixup_alpha, mixup_ratio=mixup_ratio, vae_model=vae_model,gan_model=gan_model, latent_train_loader=latent_train_loader)
        print("Evaluation on the validation set ...")
        acc, loss = evaluate(model, device, val_loader)
        print("Accuracy on the validation set: {}".format(acc))
        if hyperparameters["optim"] == "sgd":
            scheduler.step(loss)
        else:
            scheduler.step()

        print("learning rate: ", optimizer.param_groups[0]['lr'])
        print("\n")


def vae_train(vae, device, optimizer, train_loader):
    vae.train()
    train_loss = 0
    total = 0
    for batch_idx, (data, target) in enumerate(train_loader):
        data = data.to(device)
        optimizer.zero_grad()
        recon_batch, mu, log_var = vae(data)
        loss = vae_loss_function(recon_batch, data, mu, log_var, vae.x_dim)
        loss.backward()
        train_loss += loss.item()
        total += target.size(0)
        optimizer.step()
        # progress_bar(batch_idx, len(train_loader), 'Loss: %.3f' % (train_loss / total))


def vae_evaluate(vae, device, data_loader, verbose=True):
    vae.eval()
    test_loss = 0
    total = 0
    with torch.no_grad():
        for batch_idx, (data, target) in enumerate(data_loader):
            data = data.to(device)
            recon, mu, log_var = vae(data)
            total += target.size(0)
            test_loss += vae_loss_function(recon, data, mu, log_var, vae.x_dim).item()
            # if verbose:
                # progress_bar(batch_idx, len(data_loader), 'Loss: %.3f' % (test_loss / total))
    return test_loss / total


def full_vae_training(vae, train_loader, val_loader, device, hyperparameters):
    optimizer = optim.Adam(vae.parameters())
    for epoch in range(hyperparameters["vae_epochs"]):
        print("Epoch {}/{}".format(epoch, hyperparameters["vae_epochs"]))
        print("Training ...")
        vae_train(vae, device, optimizer, train_loader)
        # print("Evaluation on the validation set ...")
        # vae_evaluate(vae, device, val_loader, verbose=True)
        print("\n")
        #save a sample of the generated images
        with torch.no_grad():
            sample = torch.randn(64, vae.z_dim).to(device)
            sample = vae.decoder(sample).cpu()
            save_image(sample.view(64, 1, 28, 28), 'vae_sample_' + str(epoch) + '.png')



#trains the gan_initializer model (a standard convnet) to find the nearest latent vector to a given image
def gan_initializer_training(gan_initializer,gan:GAN):
    device = "cuda"
    batch_size = 16
    # generate batches of images with their corresponding latent vectors as the labels
    X = []
    y = []
    sample_batches = 4000
    with torch.no_grad():
        for i in range(sample_batches):
            z = torch.randn(batch_size, gan.z_dim,1,1).cuda()
            sample = gan.generator(z)
            image = sample.view(batch_size, 1, 28, 28).cpu().numpy()
            X.append(image)
            y.append(z.cpu().numpy())
            progress_bar(i, sample_batches, 'Generating samples ...')

            #save a sample
            # image = np.transpose(image, (0, 2, 3, 1))
            # fig, ax = plt.subplots(1, 5, figsize=(20, 4))
            # for i in range(5):
            #     print(z[i,:10])
            #     ax[i].imshow(image[i])
            # plt.savefig("gan_initializer_training_sample.png")

        X = np.concatenate(X, axis=0)
        y = np.concatenate(y, axis=0)

    print(X.shape, y.shape)
    X = torch.from_numpy(X).float()
    y = torch.from_numpy(y).float()
    gan_initializer.train()
    gan_initializer.cuda()
    optimizer = optim.AdamW(gan_initializer.parameters(), lr=0.001)
    train_loader = DataLoader(torch.utils.data.TensorDataset(X,y), batch_size=batch_size, shuffle=True)
    for epoch in range(6):
        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(device), target.to(device)
            optimizer.zero_grad()
            output = gan_initializer(data)
            loss = F2.mse_loss(output, target)
            loss.backward()
            optimizer.step()
            progress_bar(batch_idx, len(train_loader), 'Loss: %.3f' % loss.item())



def feature_extractor_training(feature_extractor, train_loader):
    device = "cuda"
    criterion = nn.CrossEntropyLoss()
    feature_extractor.train()
    feature_extractor.cuda()
    optimizer = optim.Adam(feature_extractor.parameters(), lr=0.001)
    for epoch in range(5):
        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(device), target.to(device)
            optimizer.zero_grad()
            output = feature_extractor(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
            progress_bar(batch_idx, len(train_loader), 'Loss: %.3f' % loss.item())



def full_gan_training(gan:GAN, train_loader, device, hyperparameters):
    G = gan.generator.to(device)

    D = gan.discriminator.to(device)

    epochs = hyperparameters["gan_epochs"]
    z_dim = hyperparameters["gan_z_dim"]

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

    G_opt = optim.AdamW(G.parameters(), lr=0.0003, betas=(0.5, 0.999))
    D_opt = optim.AdamW(D.parameters(), lr=0.0003, betas=(0.5, 0.999))

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
            
            latent = torch.randn(D_out_real.shape[0], z_dim, 1, 1, device=device)
            
            
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


    for epoch in range(epochs):
        dl, gl = train_epoch(G_opt, D_opt, G, D, train_loader)
        print(f"Epoch: {epoch}, D_loss: {dl}, G_loss: {gl}")
        with torch.no_grad():
            test_z = Variable(torch.randn(128, z_dim, 1, 1).to(device))
            generated = G(test_z)
            save_image(generated, './gan_sample_' + str(epoch) + '.png')






def score_report(model, device, val_loader, test_loader, blurred_test_loader):
    report = [['type', 'accuracy', 'loss', 'deep_fool']]
    print("Final report :")
    val_accuracy, val_loss = evaluate(model, device, val_loader, verbose=False)
    val_df_score = deepfool_score(model, device, val_loader)
    report.append(["val", val_accuracy, val_loss, val_df_score])
    print("\tPerformance on the validation set - acc. : {:0.4f}, loss : {:.4f}, DeepFool score : {:.4f}".format(val_accuracy, val_loss, val_df_score))
    test_accuracy, test_loss = evaluate(model, device, test_loader, verbose=False)
    test_df_score = deepfool_score(model, device, test_loader)
    report.append(["test", test_accuracy, test_loss, test_df_score])
    print("\tPerformance on the testing set - acc. : {:0.4f}, loss : {:.4f}, DeepFool score : {:.4f}".format(test_accuracy, test_loss, test_df_score))
    blurred_test_accuracy, blurred_test_loss = evaluate(model, device, blurred_test_loader, verbose=False)
    report.append(["blurred_test", blurred_test_accuracy, blurred_test_loss, float('nan')])
    print("\tPerformance on the blurred testing set - acc. : {:0.4f}, loss : {:.4f}".format(blurred_test_accuracy, blurred_test_loss))
    return report


def deepfool_score(model, device, test_loader):
    # return 0  # This is only to speed up testing
    model.softmax = nn.Identity()
    # test the model on adversarial examples
    norms = []
    batches_done = 0
    for batch_idx, (data, target) in enumerate(test_loader):
        data, target = data.to(device), target.to(device)
        for image, t in zip(data, target):
            t = t.unsqueeze(0)
            # generate adversarial examples
            data = deepfool(image, model, num_classes=10, overshoot=0.02, max_iter=50)
            minimal_perturbation = data[0]
            # calculate the norm of the perturbation
            norm = np.linalg.norm(minimal_perturbation)
            norms.append(norm)
        batches_done += 1
        if batches_done % 50 == 0:
            break
    return np.mean(norms)
