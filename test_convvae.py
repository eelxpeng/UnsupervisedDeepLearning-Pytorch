import torch
import torch.utils.data
from torchvision import datasets, transforms
import numpy as np
from udlp.autoencoder.convVAE import ConvVAE
import argparse

parser = argparse.ArgumentParser(description='VAE MNIST Example')
parser.add_argument('--lr', type=float, default=0.0001, metavar='N',
                    help='learning rate for training (default: 0.001)')
parser.add_argument('--batch-size', type=int, default=128, metavar='N',
                    help='input batch size for training (default: 128)')
parser.add_argument('--epochs', type=int, default=10, metavar='N',
                    help='number of epochs to train (default: 10)')
parser.add_argument('--save', type=str, default="", metavar='N',
                    help='number of epochs to train (default: 10)')
args = parser.parse_args()

train_loader = torch.utils.data.DataLoader(
    datasets.SVHN('./dataset/svhn', split='train', download=True,
                   transform=transforms.ToTensor()),
    batch_size=args.batch_size, shuffle=True, num_workers=2)
test_loader = torch.utils.data.DataLoader(
    datasets.SVHN('./dataset/svhn', split='test', download=True, transform=transforms.ToTensor()),
    batch_size=args.batch_size, shuffle=True, num_workers=2)

vae = ConvVAE(width=32, height=32, nChannels=3, hidden_size=500, z_dim=100, binary=True,
        nFilters=64)
vae.fit(train_loader, test_loader, lr=args.lr, num_epochs=args.epochs)
if args.save!="":
	torch.save(vae.state_dict(), args.save)
