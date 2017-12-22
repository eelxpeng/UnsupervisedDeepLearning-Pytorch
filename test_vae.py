import torch
import torch.utils.data
from torchvision import datasets, transforms
import numpy as np
from udlp.autoencoder.vae import VAE
import argparse

parser = argparse.ArgumentParser(description='VAE MNIST Example')
parser.add_argument('--lr', type=float, default=0.001, metavar='N',
                    help='learning rate for training (default: 0.001)')
parser.add_argument('--batch-size', type=int, default=128, metavar='N',
                    help='input batch size for training (default: 128)')
parser.add_argument('--epochs', type=int, default=10, metavar='N',
                    help='number of epochs to train (default: 10)')
args = parser.parse_args()

train_loader = torch.utils.data.DataLoader(
    datasets.MNIST('./dataset', train=True, download=True,
                   transform=transforms.ToTensor()),
    batch_size=args.batch_size, shuffle=True, num_workers=2)
test_loader = torch.utils.data.DataLoader(
    datasets.MNIST('./dataset', train=False, transform=transforms.ToTensor()),
    batch_size=args.batch_size, shuffle=True, num_workers=2)

vae = VAE(input_dim=784, z_dim=20, binary=True,
        encodeLayer=[400], decodeLayer=[400])
vae.fit(train_loader, test_loader, lr=args.lr, num_epochs=args.epochs)