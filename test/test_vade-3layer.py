import sys
sys.path.append("..")
import torch
import torch.utils.data
from torchvision import datasets, transforms
import numpy as np
from udlp.clustering.vade import VaDE
import argparse

parser = argparse.ArgumentParser(description='VAE MNIST Example')
parser.add_argument('--lr', type=float, default=0.002, metavar='N',
                    help='learning rate for training (default: 0.001)')
parser.add_argument('--batch-size', type=int, default=100, metavar='N',
                    help='input batch size for training (default: 128)')
parser.add_argument('--epochs', type=int, default=100, metavar='N',
                    help='number of epochs to train (default: 10)')
parser.add_argument('--pretrain', type=str, default="model/pretrained_vade-3layer.pt", metavar='N',
                    help='number of epochs to train (default: 10)')
parser.add_argument('--save', type=str, default="", metavar='N',
                    help='number of epochs to train (default: 10)')
args = parser.parse_args()

train_loader = torch.utils.data.DataLoader(
    datasets.MNIST('../data/mnist', train=True, download=True,
                   transform=transforms.ToTensor()),
    batch_size=args.batch_size, shuffle=True, num_workers=2)
test_loader = torch.utils.data.DataLoader(
    datasets.MNIST('../data/mnist', train=False, transform=transforms.ToTensor()),
    batch_size=args.batch_size, shuffle=True, num_workers=2)

vade = VaDE(input_dim=784, z_dim=10, n_centroids=10, binary=True,
        encodeLayer=[500,500,2000], decodeLayer=[2000,500,500])
if args.pretrain != "":
    print("Loading model from %s..." % args.pretrain)
    vade.load_model(args.pretrain)
print("Initializing through GMM..")
vade.initialize_gmm(train_loader)
vade.fit(train_loader, test_loader, lr=args.lr, batch_size=args.batch_size, num_epochs=args.epochs, anneal=True)
if args.save != "":
	vade.save_model(args.save)