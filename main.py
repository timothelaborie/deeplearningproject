from torch import optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
from model import get_vae, CifarResNet, get_standard_model,get_gan,get_gan_initializer,get_feature_extractor
from train_test import full_training, full_vae_training, score_report, evaluate,full_gan_training,gan_initializer_training,feature_extractor_training
from utils import *
import torch
import argparse
import numpy as np
import models
from torch.utils.data import random_split
from torch.utils.data.dataset import Subset
from torchvision.utils import save_image
from torch.autograd import Variable
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

display = False

# Very important : check whether DeepFool is evaluated or not

# Hyperparameter that affect the training of the different variants
RELEVANT_HYPERPARAMETER_NAMES = {
    "standard": ["epochs", "batch_size", "learning_rate", "random_seed", "momentum", "optim", "weight_decay", "gamma", "augment"],
    "mixup": ["epochs", "batch_size", "learning_rate", "mixup_alpha", "mixup_ratio", "random_seed", "momentum", "optim", "weight_decay", "gamma", "augment"],
    "manifold_mixup": ["epochs", "batch_size", "learning_rate", "mixup_alpha", "mixup_ratio", "random_seed", "momentum", "optim", "weight_decay", "gamma", "augment"],
    "mixup_vae": ["epochs", "batch_size", "learning_rate", "mixup_alpha", "mixup_ratio", "random_seed", "momentum", "optim", "weight_decay", "gamma", "vae_epochs", "vae_sharp", "vae_h_dim1", "vae_h_dim2", "vae_z_dim", "vae_lat_opt_steps", "augment"],
    "mixup_gan": ["epochs", "batch_size", "learning_rate", "mixup_alpha", "mixup_ratio", "random_seed", "momentum", "optim", "weight_decay", "gamma", "gan_z_dim", "gan_lat_opt_steps", "gan_epochs", "augment"],
}

DATASETS = ["mnist", "fashionmnist", "cifar10", "cifar100"]
VARIANTS = ["standard", "mixup", "manifold_mixup", "mixup_vae", "mixup_gan"]
OPTIMS = ["sgd", "adam"]
AUGS = ["none", "cifar10"]

parser = argparse.ArgumentParser(description="Experiment for the DeepLearning project")


# Utility commands
parser.add_argument('--download', '-d', action='store_true', help='only downloads the datasets')
parser.add_argument('--results', '-r', action='store_true', help='show the results stored on disk')

parser.add_argument('--dataset', choices=DATASETS, default="mnist", help="dataset to run the experiment on")
parser.add_argument('--variant', choices=VARIANTS, default="standard", help="training and model variant used")
parser.add_argument('--pretrained', type=int, default=0, help="use cifar10 pre-trained model")
parser.add_argument('--epochs', type=int, default=180, help="total epochs to run")
parser.add_argument('--batch_size', type=int, default=16, help="batch size")
parser.add_argument('--learning_rate', type=float, default=0.1, help="learning rate")
parser.add_argument('--mixup_alpha', type=float, default=1.0, help="parameter of the Beta distribution to sample lambda for mixup")
parser.add_argument('--mixup_ratio', type=float, default=1.0, help="ratio of the mixed up data used to train the classifier, the rest is the standard dataset")
parser.add_argument('--random_seed', type=int, default=0, help="random seed")
parser.add_argument('--vae_epochs', type=int, default=8, help="total epochs to train the VAE")
parser.add_argument('--vae_sharp', type=int, default=1, help="sharpening coefficient for the output of the VAE")
parser.add_argument('--vae_h_dim1', type=int, default=512, help="first hidden dimension of the VAE")
parser.add_argument('--vae_h_dim2', type=int, default=512, help="second hidden dimension of the VAE")
parser.add_argument('--vae_z_dim', type=int, default=16, help="dimension of the latent code for the VAE")
parser.add_argument('--vae_lat_opt_steps', type=int, default=0, help="number of steps of the optimization of the latent codes of the VAE")
parser.add_argument('--gan_epochs', type=int, default=8, help="total epochs to train the GAN")
parser.add_argument('--gan_z_dim', type=int, default=1024, help="dimension of the latent code for the GAN")
parser.add_argument('--gan_lat_opt_steps', type=int, default=1000, help="number of steps of the optimization of the latent codes of the GAN")
parser.add_argument('--momentum', type=float, default=0.9, help="momentum for SGD")
parser.add_argument('--weight_decay', type=float, default=1e-4, help="weight decay for SGD")
parser.add_argument('--gamma', type=float, default=0.1, help="lr factor")
parser.add_argument('--optim', choices=OPTIMS, default="sgd", help="optimizer")
parser.add_argument('--augment', choices=AUGS, default="none", help="additional augmentation")
args = parser.parse_args()


if args.results:
    report_results()
    exit()

if args.download:
    print("Downloading all the datasets")
    for dataset_name in DATASETS_CALLS.keys():
        get_dataset(dataset_name)
    print("Successfully stored all the datasets on disk")
    exit()


dataset_name = args.dataset
variant = args.variant

create_directory("./augmented_datasets")
create_directory("./augmented_datasets/{}".format(dataset_name))

for t in ["results", "models"]:
    create_directory("./{}".format(t))
    create_directory("./{}/{}".format(t, dataset_name))
    create_directory("./{}/{}/{}".format(t, dataset_name, variant))

experiment_name = []

relevant_hyperparameters = {}

print("Experiment on dataset {} with {} variant".format(dataset_name, variant))
print("Relevant hyper-parameters : ")
arguments = dict()
for k in vars(args):
    arguments[k] = vars(args)[k]
    if k in RELEVANT_HYPERPARAMETER_NAMES[variant]:
        print("\t{} : {}".format(k.capitalize().replace("_", " "), vars(args)[k]))

relevant_hyperparameters = mask_dict(arguments, RELEVANT_HYPERPARAMETER_NAMES[variant])

experiment_name = hyperparameter_to_name(relevant_hyperparameters)
report_file_name = "./results/{}/{}/{}.csv".format(dataset_name, variant, experiment_name)

if file_exists(report_file_name):
    print("Experiment has already been executed")
    exit()

'''
Default values : 
For MNIST : batch_size=16, learning_rate=0.001
'''

torch.manual_seed(relevant_hyperparameters["random_seed"])

device_name = "cuda" if torch.cuda.is_available() else "cpu"
print("Device : {}".format(device_name.upper()))
device = torch.device(device_name)

# Load the datasets

train_dataset, val_dataset, test_dataset = get_dataset(dataset_name, hyperparameters=relevant_hyperparameters)
_, _, blurred_test_dataset = get_dataset(dataset_name, blur=True)

# Create the loaders
train_loader = DataLoader(train_dataset, batch_size=relevant_hyperparameters["batch_size"], shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=100, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=100, shuffle=False)
blurred_test_loader = DataLoader(blurred_test_dataset, batch_size=100, shuffle=False)

if variant == "standard":
    print("\n\nTraining with the standard model")
    model = get_standard_model(dataset_name, device,args.pretrained).to(device)
    full_training(model, train_loader, val_loader, dataset_name, relevant_hyperparameters, device)
elif variant == "mixup":
    print("\n\nTraining with Mixup")
    model = get_standard_model(dataset_name, device,args.pretrained).to(device)
    full_training(model, train_loader, val_loader, dataset_name, relevant_hyperparameters, device, specificity="mixup", mixup_alpha=relevant_hyperparameters["mixup_alpha"], mixup_ratio=relevant_hyperparameters["mixup_ratio"])
elif variant == "manifold_mixup":
    print("\n\nTraining with Manifold Mixup")
    model = get_standard_model(dataset_name, device,args.pretrained).to(device)
    full_training(model, train_loader, val_loader, dataset_name, relevant_hyperparameters, device, specificity="manifold_mixup", mixup_alpha=relevant_hyperparameters["mixup_alpha"], mixup_ratio=relevant_hyperparameters["mixup_ratio"])



elif variant == "mixup_vae":
    print("\n\nTraining with a VAE")
    vae_relevant_hyperparameters = mask_dict(relevant_hyperparameters, ["vae_h_dim1", "vae_h_dim2", "vae_z_dim", "vae_epochs", "random_seed"])
    vae_hyperparameter_name = hyperparameter_to_name(vae_relevant_hyperparameters)
    vae_latent_code_relevant_hyperparameters = mask_dict(relevant_hyperparameters, ["vae_h_dim1", "vae_h_dim2", "vae_z_dim", "vae_epochs", "random_seed", "vae_lat_opt_steps"])
    vae_latent_code_hyperparameter_name = hyperparameter_to_name(vae_latent_code_relevant_hyperparameters)
    vae_file_name = "./models/{}/mixup_vae/{}.pth".format(dataset_name, vae_hyperparameter_name)
    vae_latent_code_file_name = "./augmented_datasets/{}/{}_train_latent_codes.npy".format(dataset_name, vae_latent_code_hyperparameter_name)
    vae_latent_labels_file_name = "./augmented_datasets/{}/{}_train_latent_labels.npy".format(dataset_name, vae_latent_code_hyperparameter_name)
    # Get the VAE model
    vae_model = get_vae(dataset_name, h_dim1=relevant_hyperparameters["vae_h_dim1"], h_dim2=relevant_hyperparameters["vae_h_dim2"], z_dim=relevant_hyperparameters["vae_z_dim"])
    vae_model = vae_model.to(device)
    # Train it or load it from disk if possible
    if file_exists(vae_file_name):
        print("VAE model is already on disk")
        vae_model.load_state_dict(torch.load(vae_file_name))
    else:
        print("Training the VAE model")
        full_vae_training(vae_model, train_loader, val_loader, device, relevant_hyperparameters)
        torch.save(vae_model.state_dict(), vae_file_name)
    # Generate the latent codes for each image of the training set or load the codes if possible
    if file_exists(vae_latent_code_file_name) and file_exists(vae_latent_labels_file_name):
        print("Latent codes of the training set are already stored")
        latent_x = np.load(vae_latent_code_file_name)
        latent_y = np.load(vae_latent_labels_file_name)
    else:
        print("Computing the latent codes of the training set")
        latent_x, latent_y = [], []
        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(device), target.to(device)
            flat_data = data.view(-1, vae_model.x_dim)
            latent_codes = vae_model.encoder(flat_data)[0]  # Take only the mean into consideration
            new_images = vae_model.decoder(latent_codes)
            latent_codes = torch.from_numpy(latent_codes.cpu().detach().numpy()).float().to(device)
            opt_latent_codes = latent_codes.clone().detach().requires_grad_(True)
            optimizer = optim.Adam([opt_latent_codes], lr=0.1)
            # Optimize the latent codes in order to reconstruct the image more precisely
            for i in range(vae_latent_code_relevant_hyperparameters["vae_lat_opt_steps"]):
                optimizer.zero_grad()
                gen = vae_model.decoder(opt_latent_codes).view(data.shape[0], DATASET_IMAGE_CHN[dataset_name], DATASET_IMAGE_DIM[dataset_name], DATASET_IMAGE_DIM[dataset_name])
                loss = F.mse_loss(gen, data)
                loss.backward()
                optimizer.step()
            latent_x.append(opt_latent_codes.cpu().detach().numpy())
            latent_y.append(target.cpu().detach().numpy())
            # progress_bar(batch_idx, len(train_loader))
        latent_x = np.concatenate(latent_x, axis=0)
        latent_y = np.concatenate(latent_y, axis=0)
        # Store the latent codes for later reuse
        np.save(vae_latent_code_file_name, latent_x)
        np.save(vae_latent_labels_file_name, latent_y)

    if True:
        data, target = next(iter(train_loader))
        flat_data = data.view(-1, vae_model.x_dim)
        latent_codes = vae_model.encoder(flat_data)[0]  # .cpu().detach().numpy()
        new_images = vae_model.decoder(latent_codes)
        store_png_images(data, "./original.png", dataset_name)
        store_png_images(new_images, "./reconstructed.png", dataset_name)

    latent_x = torch.from_numpy(latent_x).float()
    latent_y = torch.from_numpy(latent_y)
    latent_train_loader = DataLoader(torch.utils.data.TensorDataset(latent_x, latent_y), batch_size=relevant_hyperparameters["batch_size"], shuffle=True)
    model = get_standard_model(dataset_name, device,args.pretrained).to(device)
    # Train the model with the latent codes
    full_training(model, train_loader, val_loader, dataset_name, relevant_hyperparameters, device, specificity="mixup_vae", mixup_alpha=relevant_hyperparameters["mixup_alpha"], mixup_ratio=relevant_hyperparameters["mixup_ratio"], vae_model=vae_model, latent_train_loader=latent_train_loader)



elif variant == "mixup_gan":
    if dataset_name.endswith("mnist"):
        print("\n\nTraining with a GAN")
        gan_relevant_hyperparameters = mask_dict(relevant_hyperparameters, ["gan_z_dim", "gan_epochs", "random_seed"])
        gan_hyperparameter_name = hyperparameter_to_name(gan_relevant_hyperparameters)
        gan_latent_code_relevant_hyperparameters = mask_dict(relevant_hyperparameters, ["gan_z_dim", "gan_epochs", "random_seed", "gan_lat_opt_steps"])
        gan_latent_code_hyperparameter_name = hyperparameter_to_name(gan_latent_code_relevant_hyperparameters)
        gan_file_name = "./models/{}/mixup_gan/{}.pth".format(dataset_name, gan_hyperparameter_name)
        gan_initializer_file_name = "./models/{}/mixup_gan/{}_initializer.pth".format(dataset_name, gan_hyperparameter_name)
        feature_extractor_file_name = "./models/{}/mixup_gan/feature_extractor.pth".format(dataset_name)
        gan_latent_code_file_name = "./augmented_datasets/{}/{}_train_latent_codes.npy".format(dataset_name, gan_latent_code_hyperparameter_name)
        gan_latent_labels_file_name = "./augmented_datasets/{}/{}_train_latent_labels.npy".format(dataset_name, gan_latent_code_hyperparameter_name)
        # Get the GAN model
        gan_model = get_gan(z_dim=relevant_hyperparameters["gan_z_dim"])
        gan_model = gan_model.to(device)
        # Train it or load it from disk if possible
        if file_exists(gan_file_name):
            print("GAN model is already on disk")
            gan_model.generator.load_state_dict(torch.load(gan_file_name))
        else:
            print("Training the GAN model")
            full_gan_training(gan_model, train_loader, device, relevant_hyperparameters)
            torch.save(gan_model.generator.state_dict(), gan_file_name)

        if display:
            with torch.no_grad():
                test_z = Variable(torch.randn(128, 1024, 1, 1).to(device))
                generated = gan_model.generator(test_z)
                save_image(generated, './gan.png')

        # Generate the latent codes for each image of the training set or load the codes if possible
        if file_exists(gan_latent_code_file_name) and file_exists(gan_latent_labels_file_name):
            print("Latent codes of the training set are already stored")
            latent_x = np.load(gan_latent_code_file_name)
            latent_y = np.load(gan_latent_labels_file_name)

            if display:
                fig, ax = plt.subplots(3, 10, figsize=(10, 2))
                for i in range(10):
                    ax[0, i].imshow(gan_model.generator(torch.from_numpy(latent_x[i][None]).cuda()).cpu().detach().numpy()[0].transpose(1,2,0))
                    ax[1, i].imshow(gan_model.generator(torch.from_numpy(latent_x[i + 10][None]).cuda()).cpu().detach().numpy()[0].transpose(1,2,0))
                    ax[2, i].imshow(gan_model.generator(torch.from_numpy(latent_x[i + 20][None]).cuda()).cpu().detach().numpy()[0].transpose(1,2,0))
                plt.show()

        else:
            print("Computing the latent codes of the training set")
            gan_initializer = get_gan_initializer(relevant_hyperparameters["gan_z_dim"]).cuda()
            feature_extractor = get_feature_extractor().cuda()
            if file_exists(gan_initializer_file_name):
                print("gan_initializer model is already on disk")
                gan_initializer.load_state_dict(torch.load(gan_initializer_file_name))
            else:
                print("Training the gan_initializer model")
                gan_initializer_training(gan_initializer,gan_model)
                torch.save(gan_initializer.state_dict(), gan_initializer_file_name)

            if file_exists(feature_extractor_file_name):
                print("feature_extractor model is already on disk")
                feature_extractor.load_state_dict(torch.load(feature_extractor_file_name))
            else:
                print("Training the feature_extractor model")
                feature_extractor_training(feature_extractor,train_loader)
                torch.save(feature_extractor.state_dict(), feature_extractor_file_name)


            latent_x, latent_y = [], []
            inversion_loader = DataLoader(train_dataset, batch_size=16, shuffle=False)
            for batch_idx, (data, target) in enumerate(inversion_loader):
                data, target = data.to(device), target.to(device)
                latent_codes = gan_initializer(data)
                # latent_codes = torch.randn(relevant_hyperparameters["batch_size"], gan_model.z_dim,1,1).cuda()
                # latent_codes = torch.ones(relevant_hyperparameters["batch_size"], gan_model.z_dim,1,1).cuda()
                new_images = gan_model.generator(latent_codes)
                latent_codes = torch.from_numpy(latent_codes.cpu().detach().numpy()).float().to(device)
                opt_latent_codes = latent_codes.clone().detach().requires_grad_(True)
                # optimizer = optim.Adam([opt_latent_codes], lr=0.001)
                optimizer = torch.optim.AdamW([opt_latent_codes], betas=(0.95, 0.999), lr=3e-2)

                # plot the original images, the initialized images and the reconstructed images
                if display:
                    fig, ax = plt.subplots(3, 10, figsize=(10, 2))
                    with torch.no_grad():
                        for i in range(10):
                            ax[0][i].imshow(data[i].cpu().detach().numpy().transpose(1,2,0),cmap='Greys',  interpolation='nearest')
                            ax[1][i].imshow(gan_model.generator(opt_latent_codes)[i].cpu().detach().numpy().transpose(1,2,0),cmap='Greys',  interpolation='nearest')

                # Optimize the latent codes in order to reconstruct the image more precisely
                for i in range(gan_latent_code_relevant_hyperparameters["gan_lat_opt_steps"]):
                    optimizer.zero_grad()
                    gen = gan_model.generator(opt_latent_codes)
                    loss = F.mse_loss(feature_extractor.extract_features((gen+1)/2), feature_extractor.extract_features(data))
                    # loss = F.mse_loss((gen+1)/2, data)
                    loss.backward()
                    optimizer.step()
                latent_x.append(opt_latent_codes.cpu().detach().numpy())
                latent_y.append(target.cpu().detach().numpy())
                progress_bar(batch_idx, len(inversion_loader), "Computing latent codes, batch: " + str(batch_idx) + "/" + str(len(inversion_loader)))

                if display:
                    with torch.no_grad():
                        for i in range(10):
                            img1 = data[i].cpu().detach().numpy().transpose(1,2,0)
                            img2 = gan_model.generator(opt_latent_codes)[i].cpu().detach().numpy().transpose(1,2,0)
                            img2 += 1
                            img2 /= 2
                            ax[2][i].imshow(img2,cmap='Greys',  interpolation='nearest')
                        plt.show()

                # if batch_idx == 0:
                #     break

            latent_x = np.concatenate(latent_x, axis=0)
            latent_y = np.concatenate(latent_y, axis=0)
            # Store the latent codes for later reuse
            np.save(gan_latent_code_file_name, latent_x)
            np.save(gan_latent_labels_file_name, latent_y)
            print("")
            print("latent_x.shape", latent_x.shape)
            print("latent_y.shape", latent_y.shape)
            
 

        latent_x = torch.from_numpy(latent_x).float()
        latent_y = torch.from_numpy(latent_y).long()

        print("latent_x.shape", latent_x.shape)
        print("latent_y.shape", latent_y.shape)

        mse_list = []
        # # filter out images so only the best half are used
        print("filtering latent codes")
        latent_x_filtered = []
        latent_y_filtered = []
        median = 0.16574298590421677
        with torch.no_grad():
            temp_loader = DataLoader(train_dataset, batch_size=1, shuffle=False)
            i = 0
            for (z,y,img) in zip(latent_x.cuda(),latent_y.cuda(), temp_loader):
                # fig, ax = plt.subplots(2, 1, figsize=(10, 2))
                img = img[0].cuda()
                #check mse between img and generator(z)
                recon = gan_model.generator(z.unsqueeze(0))
                mse = F.mse_loss(img, recon)
                if mse.item() < median:
                    latent_x_filtered.append(z.cpu().detach().numpy())
                    latent_y_filtered.append(y.cpu().detach().numpy())
                mse_list.append(mse.item())
                # ax[0].imshow(img.cpu().detach().numpy()[0].transpose(1,2,0),cmap='Greys',  interpolation='nearest')
                # ax[1].imshow(recon.cpu().detach().numpy()[0].transpose(1,2,0),cmap='Greys',  interpolation='nearest')
                # plt.show()
                i+=1
                # if i == 100:
                #     break


        mse_list = np.array(mse_list)
        mse_list = np.sort(mse_list)
        # plt.plot(mse_list)
        # plt.show()
        print("mse_list.min()", mse_list.min(), "mse_list.max()", mse_list.max(), "mse_list.mean()", mse_list.mean(), "mse_list.median()", np.median(mse_list))
           

        latent_x_filtered = torch.from_numpy(np.array(latent_x_filtered)).float()
        latent_y_filtered = torch.from_numpy(np.array(latent_y_filtered)).long()

        

        print("latent_x_filtered.shape", latent_x_filtered.shape)
        print("latent_y_filtered.shape", latent_y_filtered.shape)

        latent_train_loader = DataLoader(torch.utils.data.TensorDataset(latent_x_filtered, latent_y_filtered), batch_size=relevant_hyperparameters["batch_size"], shuffle=True)
        model = get_standard_model(dataset_name, device,args.pretrained).to(device)
        # Train the model with the latent codes
        full_training(model, train_loader, val_loader, dataset_name, relevant_hyperparameters, device, specificity="mixup_gan", mixup_alpha=relevant_hyperparameters["mixup_alpha"], mixup_ratio=relevant_hyperparameters["mixup_ratio"], gan_model=gan_model, latent_train_loader=latent_train_loader)

    else:
        latent = np.load("/cluster/home/bgunders/dl_inversion_data/grad_latents_00000_50000.npy", allow_pickle=True).item()
        latent_x = []
        latent_y = []
        for (key,value) in latent.items():
            latent_x.append(value)
            latent_y.append(int(key[6]))

        latent_x = torch.from_numpy(np.array(latent_x)).float()
        var1, var2 = random_split(latent_x, [40000, 10000], generator=torch.Generator().manual_seed(42))
        latent_x = latent_x[var1.indices]
        latent_y = torch.from_numpy(np.array(latent_y)[var1.indices])
        latent_y = latent_y.long()
        latent_train_loader = DataLoader(torch.utils.data.TensorDataset(latent_x, latent_y), batch_size=relevant_hyperparameters["batch_size"], shuffle=True)
        model = get_standard_model(dataset_name, device,args.pretrained).to(device)
        from sg3 import SG3Generator
        gan_model = SG3Generator(checkpoint_path='/cluster/home/bgunders/dl_inversion_data/sg2c10-32.pkl').decoder.eval().cuda()

        #save a sample
        if display:
            latent_it = iter(latent_train_loader)
            latent_data, latent_target = next(latent_it)
            inputs, targets_a, targets_b, lam = mixup_data(latent_data, latent_target, device=device, alpha=1.0)
            print(inputs.shape)
            inputs = inputs.cuda()
            imgs = gan_model.synthesis(inputs, noise_mode='const', force_fp32=True)
            imgs = imgs.cpu().detach().numpy()
            imgs = np.transpose(imgs, (0, 2, 3, 1))
            fig, ax = plt.subplots(1, 5, figsize=(20, 4))
            for i in range(5):
                print(imgs[i].min(), imgs[i].max())
                ax[i].imshow(imgs[i])
                print("targets_a",targets_a[i], "targets_b",targets_b[i], "lam",lam)
            plt.savefig("sample1.png")
            #save a sample of the original
            it = iter(train_loader)
            data, target = next(it)
            imgs = data.cpu().detach().numpy()
            imgs = np.transpose(imgs, (0, 2, 3, 1))
            fig, ax = plt.subplots(1, 5, figsize=(20, 4))
            for i in range(5):
                print(imgs[i].min(), imgs[i].max())
                ax[i].imshow(imgs[i])
            plt.savefig("sample2.png")


        # Train the model with the latent codes
        full_training(model, train_loader, val_loader, dataset_name, relevant_hyperparameters, device, specificity="mixup_gan", mixup_alpha=relevant_hyperparameters["mixup_alpha"], mixup_ratio=relevant_hyperparameters["mixup_ratio"], gan_model=gan_model, latent_train_loader=latent_train_loader)
else:
    assert False

report = score_report(model, device, val_loader, test_loader, blurred_test_loader)
write_csv(report_file_name, report)
