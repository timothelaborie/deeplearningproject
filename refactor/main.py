from model import get_model
from train_test import full_training
from utils import *
import torch
import argparse

"""
To do : - Add the Manifold-Mixup baseline
        - Include GAN and VAE
        - Find good networks in order to achieve near-SOTA accuracy on the performance
"""

parser = argparse.ArgumentParser()

parser.add_argument('--dataset', choices=["mnist", "fashionmnist", "cifar10", "cifar100"], default="mnist")
parser.add_argument('--epochs', type=int, default=10)
parser.add_argument('--batch_size', type=int, default=16)
parser.add_argument('--learning_rate', type=float, default=0.001)
# parser.add_argument('--other', nargs='+', type=str, default=[])

args = parser.parse_args()

print("Program arguments : ")
for k in vars(args):
    print("\t{} : {}".format(k.capitalize().replace("_", " "), vars(args)[k]))

'''
Default values : 
For MNIST : batch_size=16, learning_rate=0.001
'''

if not dir_exists("./datasets"):
    download_datasets()

dataset_name = args.dataset
epochs = args.epochs
batch_size = args.batch_size
learning_rate = args.learning_rate

device_name = "cuda" if torch.cuda.is_available() else "cpu"
print("Device : {}".format(device_name.upper()))
device = torch.device(device_name)
torch.manual_seed(0)

blur_images(dataset_name)

print("\n\nTraining with the standard dataset")
train_dataset = load_dataset(dataset_name, train=True)
test_dataset = load_dataset(dataset_name, train=False)
model = get_model(dataset_name)().to(device)
full_training(model, train_dataset, test_dataset, epochs, batch_size, learning_rate, device)

print("\n\nTraining with the blurred dataset")
blurred_train_dataset = load_dataset(dataset_name, train=True, specificity="blurred")
test_dataset = load_dataset(dataset_name, train=False)
model = get_model(dataset_name)().to(device)
full_training(model, blurred_train_dataset, test_dataset, epochs, batch_size, learning_rate, device)

print("\n\nTraining with the Mixup dataset")
train_dataset = load_dataset(dataset_name, train=True)
test_dataset = load_dataset(dataset_name, train=False)
model = get_model(dataset_name)().to(device)
full_training(model, train_dataset, test_dataset, epochs, batch_size, learning_rate, device, specificity="mixup")
