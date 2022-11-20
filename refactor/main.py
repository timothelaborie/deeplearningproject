from model import get_model
from train_test import full_training, evaluate, score_report
from utils import *
import torch
import argparse

"""
To do :
        - Add the Manifold-Mixup baseline
        - Include GAN
        - Include VAE
        - Find good networks in order to achieve near-SOTA accuracy on the standard variant
        - Cache the intermediate models when necessary
        - Add a command to read all the previous results and return the best hyperparameters
"""
RELEVANT_HYPERPARAMETER_NAMES = {
    "standard": ["epochs", "batch_size", "learning_rate", "random_seed"],
    "mixup": ["epochs", "batch_size", "learning_rate", "random_seed"],
}

parser = argparse.ArgumentParser(description="Experiment for the DeepLearning project")

parser.add_argument('--download_only', '-d', action='store_true', help='only downloads the datasets')
parser.add_argument('--dataset', choices=["mnist", "fashionmnist", "cifar10", "cifar100"], default="mnist", help="dataset to run the experiment on")
parser.add_argument('--variant', choices=["standard", "mixup"], default="standard", help="training and model variant used")
parser.add_argument('--epochs', type=int, default=10, help="total epochs to run")
parser.add_argument('--batch_size', type=int, default=16, help="batch size")
parser.add_argument('--learning_rate', type=float, default=0.001, help="learning rate")
parser.add_argument('--random_seed', type=int, default=0, help="random seed")
# parser.add_argument('--other', nargs='+', type=str, default=[])

args = parser.parse_args()

if args.download_only:
    print("Downloading all the datasets")
    for dataset_name in DATASETS_CALLS.keys():
        download_dataset(dataset_name)
        blur_images(dataset_name)
    print("Successfully stored all the datasets on disk")
    exit()


dataset_name = args.dataset
variant = args.variant

create_directory("./results")
create_directory("./results/{}".format(dataset_name))
create_directory("./results/{}/{}".format(dataset_name, variant))

experiment_name = []

relevant_hyperparameters = {}

print("Experiment on dataset {} with {} variant".format(dataset_name, variant))

print("Relevant hyper-parameters : ")
for k in vars(args):
    if k in RELEVANT_HYPERPARAMETER_NAMES[variant]:
        relevant_hyperparameters[k] = vars(args)[k]
        print("\t{} : {}".format(k.capitalize().replace("_", " "), vars(args)[k]))

experiment_name = "&".join(sorted(["{}:{}".format(k, relevant_hyperparameters[k]) for k in relevant_hyperparameters]))
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

download_dataset(dataset_name)
blur_images(dataset_name)

device_name = "cuda" if torch.cuda.is_available() else "cpu"
print("Device : {}".format(device_name.upper()))
device = torch.device(device_name)


train_dataset, val_dataset = load_dataset(dataset_name, train_val=True)
test_dataset = load_dataset(dataset_name, train_val=False)
blurred_test_dataset = load_dataset(dataset_name, train_val=False, specificity="blurred")

train_loader = DataLoader(train_dataset, batch_size=relevant_hyperparameters["batch_size"], shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=100, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=100, shuffle=False)
blurred_test_loader = DataLoader(blurred_test_dataset, batch_size=100, shuffle=False)

if variant == "standard":
    print("\n\nTraining with the standard model")
    model_call, model_kwargs = get_model(dataset_name)
    model = model_call(**model_kwargs).to(device)
    full_training(model, train_loader, val_loader, relevant_hyperparameters, device)
elif variant == "mixup":
    print("\n\nTraining with Mixup")
    model_call, model_kwargs = get_model(dataset_name)
    model = model_call(**model_kwargs).to(device)
    full_training(model, train_loader, val_loader, relevant_hyperparameters, device, specificity="mixup")
else:
    assert False

report = score_report(model, device, val_loader, test_loader, blurred_test_loader)
write_csv(report_file_name, report)
