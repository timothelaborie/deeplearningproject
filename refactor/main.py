from torch import optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
from data import DATASETS_CALLS, download_dataset, blur_images, load_dataset, DATASET_IMAGE_CHN, DATASET_IMAGE_DIM
from model import get_standard_model, get_vae
from train_test import full_training, full_vae_training, score_report
from utils import *
import torch
import argparse

"""
To do :
        - Include GAN
        - Save the CIFAR-10 images and VAE generated images on disk to see how the VAE performs (seems to be a bug with CIFAR-10)
        - Find good networks in order to achieve near-SOTA accuracy on standard training
        - Start the hyper-parameter search
"""

# Hyperparameter that affect the training of the different variants
RELEVANT_HYPERPARAMETER_NAMES = {
    "standard": ["epochs", "batch_size", "learning_rate", "random_seed"],
    "mixup": ["epochs", "batch_size", "learning_rate", "random_seed"],
    "manifold_mixup": ["epochs", "batch_size", "learning_rate", "random_seed"],
    "vae": ["epochs", "batch_size", "learning_rate", "random_seed", "vae_epochs", "vae_h_dim1", "vae_h_dim2", "vae_z_dim", "vae_lat_opt_steps"],
}

DATASETS = ["mnist", "fashionmnist", "cifar10", "cifar100"]
VARIANTS = ["standard", "mixup", "manifold_mixup", "vae"]

parser = argparse.ArgumentParser(description="Experiment for the DeepLearning project")

# Utility commands
parser.add_argument('--download', '-d', action='store_true', help='only downloads the datasets')
parser.add_argument('--results', '-r', action='store_true', help='show the results stored on disk')

parser.add_argument('--dataset', choices=DATASETS, default="mnist", help="dataset to run the experiment on")
parser.add_argument('--variant', choices=VARIANTS, default="standard", help="training and model variant used")
parser.add_argument('--epochs', type=int, default=10, help="total epochs to run")
parser.add_argument('--batch_size', type=int, default=16, help="batch size")
parser.add_argument('--learning_rate', type=float, default=0.001, help="learning rate")
parser.add_argument('--random_seed', type=int, default=0, help="random seed")
parser.add_argument('--vae_epochs', type=int, default=8, help="total epochs to train the vae")
parser.add_argument('--vae_h_dim1', type=int, default=512, help="first hidden dimension of the VAE")
parser.add_argument('--vae_h_dim2', type=int, default=512, help="second hidden dimension of the VAE")
parser.add_argument('--vae_z_dim', type=int, default=16, help="dimension of the latent code for the VAE")
parser.add_argument('--vae_lat_opt_steps', type=int, default=0, help="number of steps of the optimization of the latent codes of the VAE")

args = parser.parse_args()

if args.results:
    report_results()
    exit()

if args.download:
    print("Downloading all the datasets")
    for dataset_name in DATASETS_CALLS.keys():
        download_dataset(dataset_name)
        blur_images(dataset_name)
    print("Successfully stored all the datasets on disk")
    exit()


dataset_name = args.dataset
variant = args.variant

# Make sure that the dataset is stored on disk
download_dataset(dataset_name)
blur_images(dataset_name)

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
    print("Experiment has already been executed with the following results :")
    read_csv(report_file_name)
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
train_dataset, val_dataset = load_dataset(dataset_name, train_val=True)
test_dataset = load_dataset(dataset_name, train_val=False)
blurred_test_dataset = load_dataset(dataset_name, train_val=False, specificity="blurred")

# Create the loaders
train_loader = DataLoader(train_dataset, batch_size=relevant_hyperparameters["batch_size"], shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=100, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=100, shuffle=False)
blurred_test_loader = DataLoader(blurred_test_dataset, batch_size=100, shuffle=False)

if variant == "standard":
    print("\n\nTraining with the standard model")
    model = get_standard_model(dataset_name, device)
    model = model.to(device)
    full_training(model, train_loader, val_loader, dataset_name, relevant_hyperparameters, device)
elif variant == "mixup":
    print("\n\nTraining with Mixup")
    model = get_standard_model(dataset_name, device)
    model = model.to(device)
    full_training(model, train_loader, val_loader, dataset_name, relevant_hyperparameters, device, specificity="mixup")
elif variant == "manifold_mixup":
    print("\n\nTraining with Manifold Mixup")
    model = get_standard_model(dataset_name, device)
    model = model.to(device)
    full_training(model, train_loader, val_loader, dataset_name, relevant_hyperparameters, device, specificity="manifold_mixup")
elif variant == "vae":
    print("\n\nTraining with a VAE")
    vae_relevant_hyperparameters = mask_dict(relevant_hyperparameters, ["vae_h_dim1", "vae_h_dim2", "vae_z_dim", "vae_epochs", "random_seed"])
    vae_hyperparameter_name = hyperparameter_to_name(vae_relevant_hyperparameters)
    vae_latent_code_relevant_hyperparameters = mask_dict(relevant_hyperparameters, ["vae_h_dim1", "vae_h_dim2", "vae_z_dim", "vae_epochs", "random_seed", "vae_lat_opt_steps"])
    vae_latent_code_hyperparameter_name = hyperparameter_to_name(vae_latent_code_relevant_hyperparameters)

    vae_file_name = "./models/{}/vae/{}.pth".format(dataset_name, vae_hyperparameter_name)
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
                gen = vae_model.decoder(opt_latent_codes).view(relevant_hyperparameters["batch_size"], DATASET_IMAGE_CHN[dataset_name], DATASET_IMAGE_DIM[dataset_name], DATASET_IMAGE_DIM[dataset_name])
                loss = F.mse_loss(gen, data)
                loss.backward()
                optimizer.step()
            latent_x.append(opt_latent_codes.cpu().detach().numpy())
            latent_y.append(target.cpu().detach().numpy())
            progress_bar(batch_idx, len(train_loader))
        latent_x = np.concatenate(latent_x, axis=0)
        latent_y = np.concatenate(latent_y, axis=0)
        # Store the latent codes for later reuse
        np.save(vae_latent_code_file_name, latent_x)
        np.save(vae_latent_labels_file_name, latent_y)
    latent_x = torch.from_numpy(latent_x).float()
    latent_y = torch.from_numpy(latent_y).float()
    latent_train_loader = DataLoader(torch.utils.data.TensorDataset(latent_x, latent_y), batch_size=relevant_hyperparameters["batch_size"], shuffle=True)
    model = get_standard_model(dataset_name, device)
    model = model.to(device)
    # Train the model with the latent codes
    full_training(model, latent_train_loader, val_loader, dataset_name, relevant_hyperparameters, device, specificity="mixup_vae", vae_model=vae_model)
else:
    assert False

report = score_report(model, device, val_loader, test_loader, blurred_test_loader)
write_csv(report_file_name, report)
