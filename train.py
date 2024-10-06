import argparse
import torch
from torchvision import models
from train_func import train_model  # Assume you have a separate module for training functions
import os
import warnings
warnings.filterwarnings("ignore", category=UserWarning)

def main():
    parser = argparse.ArgumentParser(description='Train a neural network on a dataset.')
    parser.add_argument('data_directory', type=str, help='Directory of the dataset')
    parser.add_argument('--save_dir', type=str, default='.', help='Directory to save checkpoints')
    parser.add_argument('--arch', type=str, default='vgg16', help='Architecture to use (e.g., vgg13, vgg16)')
    parser.add_argument('--learning_rate', type=float, default=0.001, help='Learning rate for training')
    parser.add_argument('--hidden_units', type=int, default=512, help='Number of hidden units in the classifier')
    parser.add_argument('--epochs', type=int, default=5, help='Number of epochs for training')
    parser.add_argument('--gpu', action='store_true', help='Use GPU for training')

    args = parser.parse_args()

    # Check if GPU is available
    device = torch.device("cuda" if args.gpu and torch.cuda.is_available() else "cpu")

    # Load data, model, and train the model
    # model = models.__dict__[args.arch](pretrained=True)  # pretrained is deprecated
    model = models.__dict__[args.arch](weights=f"{args.arch.upper()}_Weights.DEFAULT")  # Load the specified model architecture
    train_model(model, args.data_directory, args.save_dir, args.learning_rate, args.hidden_units, args.epochs, args.arch, device)

if __name__ == '__main__':
    main()
