from torch import optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
from model import get_vae, CifarResNet, get_standard_model,get_gan
from train_test import full_training, full_vae_training, score_report, evaluate,full_gan_training
from utils import *
import torch
import argparse
import numpy as np
import models
from torch.utils.data import random_split
from torch.utils.data.dataset import Subset

# Very important : check whether DeepFool is evaluated or not

# Hyperparameter that affect the training of the different variants
RELEVANT_HYPERPARAMETER_NAMES = {
    "standard": ["epochs", "batch_size", "learning_rate", "random_seed"],
    "mixup": ["epochs", "batch_size", "learning_rate", "mixup_alpha", "mixup_ratio", "random_seed"],
    "manifold_mixup": ["epochs", "batch_size", "learning_rate", "mixup_alpha", "mixup_ratio", "random_seed"],
    "mixup_vae": ["epochs", "batch_size", "learning_rate", "mixup_alpha", "mixup_ratio", "random_seed", "vae_epochs", "vae_sharp", "vae_h_dim1", "vae_h_dim2", "vae_z_dim", "vae_lat_opt_steps"],
    "mixup_gan": ["epochs", "batch_size", "learning_rate", "mixup_alpha", "mixup_ratio", "random_seed", "gan_epochs", "gan_z_dim", "gan_lat_opt_steps"],
}

DATASETS = ["mnist", "fashionmnist", "cifar10", "cifar100"]
VARIANTS = ["standard", "mixup", "manifold_mixup", "mixup_vae", "mixup_gan"]

parser = argparse.ArgumentParser(description="Experiment for the DeepLearning project")


# Utility commands
parser.add_argument('--download', '-d', action='store_true', help='only downloads the datasets')
parser.add_argument('--results', '-r', action='store_true', help='show the results stored on disk')

parser.add_argument('--dataset', choices=DATASETS, default="mnist", help="dataset to run the experiment on")
parser.add_argument('--variant', choices=VARIANTS, default="standard", help="training and model variant used")
parser.add_argument('--pretrained', type=int, default=0, help="use cifar10 pre-trained model")
parser.add_argument('--epochs', type=int, default=10, help="total epochs to run")
parser.add_argument('--batch_size', type=int, default=16, help="batch size")
parser.add_argument('--learning_rate', type=float, default=0.001, help="learning rate")
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
parser.add_argument('--gan_z_dim', type=int, default=100, help="dimension of the latent code for the GAN")
parser.add_argument('--gan_lat_opt_steps', type=int, default=5, help="number of steps of the optimization of the latent codes of the GAN")

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

train_dataset, val_dataset, test_dataset = get_dataset(dataset_name)
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

    if False:
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
        # Generate the latent codes for each image of the training set or load the codes if possible
        if file_exists(gan_latent_code_file_name) and file_exists(gan_latent_labels_file_name):
            print("Latent codes of the training set are already stored")
            latent_x = np.load(gan_latent_code_file_name)
            latent_y = np.load(gan_latent_labels_file_name)
        else:
            pass
            # print("Computing the latent codes of the training set")
            # latent_x, latent_y = [], []
            # for batch_idx, (data, target) in enumerate(train_loader):
            #     data, target = data.to(device), target.to(device)
            #     flat_data = data.view(-1, gan_model.x_dim)
            #     latent_codes = gan_model.encoder(flat_data)[0]  # Take only the mean into consideration
            #     new_images = gan_model.decoder(latent_codes)
            #     latent_codes = torch.from_numpy(latent_codes.cpu().detach().numpy()).float().to(device)
            #     opt_latent_codes = latent_codes.clone().detach().requires_grad_(True)
            #     optimizer = optim.Adam([opt_latent_codes], lr=0.1)
            #     # Optimize the latent codes in order to reconstruct the image more precisely
            #     for i in range(gan_latent_code_relevant_hyperparameters["gan_lat_opt_steps"]):
            #         optimizer.zero_grad()
            #         gen = gan_model.decoder(opt_latent_codes).view(data.shape[0], DATASET_IMAGE_CHN[dataset_name], DATASET_IMAGE_DIM[dataset_name], DATASET_IMAGE_DIM[dataset_name])
            #         loss = F.mse_loss(gen, data)
            #         loss.backward()
            #         optimizer.step()
            #     latent_x.append(opt_latent_codes.cpu().detach().numpy())
            #     latent_y.append(target.cpu().detach().numpy())
            #     # progress_bar(batch_idx, len(train_loader))
            # latent_x = np.concatenate(latent_x, axis=0)
            # latent_y = np.concatenate(latent_y, axis=0)
            # # Store the latent codes for later reuse
            # np.save(gan_latent_code_file_name, latent_x)
            # np.save(gan_latent_labels_file_name, latent_y)

        latent_x = torch.from_numpy(latent_x).float()
        latent_y = torch.from_numpy(latent_y)
        latent_train_loader = DataLoader(torch.utils.data.TensorDataset(latent_x, latent_y), batch_size=relevant_hyperparameters["batch_size"], shuffle=True)
        model = get_standard_model(dataset_name, device,args.pretrained).to(device)
        # Train the model with the latent codes
        full_training(model, train_loader, val_loader, dataset_name, relevant_hyperparameters, device, specificity="mixup_gan", mixup_alpha=relevant_hyperparameters["mixup_alpha"], mixup_ratio=relevant_hyperparameters["mixup_ratio"], gan_model=gan_model, latent_train_loader=latent_train_loader)

    else:
        latent = np.load("grad_latents_00000_50000.npy", allow_pickle=True).item()
        latent_x = []
        latent_y = []
        for (key,value) in latent.items():
            latent_x.append(value)
            latent_y.append(int(key[6]))

        latent_x = torch.from_numpy(np.array(latent_x)).float()
        var1, var2 = random_split(latent_x, [40000, 10000], generator=torch.Generator().manual_seed(42))
        var3:Subset = var1
        latent_x = latent_x[var3.indices]
        # print(latent_x.shape)
        latent_y = torch.from_numpy(np.array(latent_y)[var3.indices])
        #convert to long
        latent_y = latent_y.long()
        latent_train_loader = DataLoader(torch.utils.data.TensorDataset(latent_x, latent_y), batch_size=relevant_hyperparameters["batch_size"], shuffle=True)
        model = get_standard_model(dataset_name, device,args.pretrained).to(device)
        from sg3 import SG3Generator
        gan_model = SG3Generator(checkpoint_path='./sg2c10-32.pkl').decoder.cuda()
        # Train the model with the latent codes
        full_training(model, train_loader, val_loader, dataset_name, relevant_hyperparameters, device, specificity="mixup_gan", mixup_alpha=relevant_hyperparameters["mixup_alpha"], mixup_ratio=relevant_hyperparameters["mixup_ratio"], gan_model=gan_model, latent_train_loader=latent_train_loader)
else:
    assert False

report = score_report(model, device, val_loader, test_loader, blurred_test_loader)
write_csv(report_file_name, report)
