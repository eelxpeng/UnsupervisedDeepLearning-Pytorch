import torch
import torch.nn as nn
from torch.nn import Parameter
import torch.nn.functional as F
import torch.optim as optim
import torchvision
from torchvision import datasets, transforms
from torch.autograd import Variable
from torchvision.utils import save_image

import numpy as np
import math
# from udlp.utils import Dataset, masking_noise
# from udlp.ops import MSELoss, BCELoss

def buildEncoderNetwork(input_channels, nFilters, hidden_size):
    net = []
    net.append(nn.Conv2d(input_channels, nFilters, kernel_size=4, stride=2, padding=1))
    net.append(nn.BatchNorm2d(nFilters))
    net.append(nn.ReLU(True))

    net.append(nn.Conv2d(nFilters, 2*nFilters, kernel_size=4, stride=2, padding=1))
    net.append(nn.BatchNorm2d(2*nFilters))
    net.append(nn.ReLU(True))

    net.append(nn.Conv2d(2*nFilters, 4*nFilters, kernel_size=4, stride=2, padding=1))
    net.append(nn.BatchNorm2d(4*nFilters))
    net.append(nn.ReLU(True))

    net.append(nn.Conv2d(4*nFilters, hidden_size, kernel_size=4))
    net.append(nn.BatchNorm2d(hidden_size))
    net.append(nn.ReLU(True))

    return nn.Sequential(*net)

def buildDecoderNetwork(hidden_size, nFilters, output_channels):
    net = []
    net.append(nn.ConvTranspose2d(hidden_size, 4*nFilters, kernel_size=4))
    net.append(nn.BatchNorm2d(4*nFilters))
    net.append(nn.ReLU(True))

    net.append(nn.ConvTranspose2d(4*nFilters, 2*nFilters, kernel_size=4, stride=2, padding=1))
    net.append(nn.BatchNorm2d(2*nFilters))
    net.append(nn.ReLU(True))

    net.append(nn.ConvTranspose2d(2*nFilters, nFilters, kernel_size=4, stride=2, padding=1))
    net.append(nn.BatchNorm2d(nFilters))
    net.append(nn.ReLU(True))

    net.append(nn.ConvTranspose2d(nFilters, output_channels, kernel_size=4, stride=2, padding=1))
    return nn.Sequential(*net)

class ConvVAE(nn.Module):
    def __init__(self, width=32, height=32, nChannels=3, hidden_size=500, 
        z_dim=20, binary=True, nFilters=64):
        super(self.__class__, self).__init__()
        self.z_dim = z_dim
        self.hidden_size = hidden_size
        self.width = width
        self.height = height
        self.nChannels = nChannels
        self.encoder = buildEncoderNetwork(nChannels, nFilters, hidden_size)
        self.decoder = buildDecoderNetwork(hidden_size, nFilters, nChannels)
        self._enc_mu = nn.Linear(hidden_size, z_dim)
        self._enc_log_sigma = nn.Linear(hidden_size, z_dim)
        self._dec = nn.Linear(z_dim, hidden_size)
        self._dec_bn = nn.BatchNorm1d(hidden_size)
        self._dec_relu = nn.ReLU(True)
        self._dec_act = None
        if binary:
            self._dec_act = nn.Sigmoid()

    def reparameterize(self, mu, logvar):
        if self.training:
          std = logvar.mul(0.5).exp_()
          eps = Variable(std.data.new(std.size()).normal_())
          return eps.mul(std).add_(mu)
        else:
          return mu

    def decode(self, z):
        h = self._dec_relu(self._dec_bn(self._dec(z)))
        h = h.view(-1, self.hidden_size, 1, 1)
        x = self.decoder(h)
        x = x.view(-1, self.nChannels, self.height, self.width)
        if self._dec_act is not None:
            x = self._dec_act(x)
        return x

    def loss_function(self, recon_x, x, mu, logvar):
        recon_x = recon_x.view(recon_x.size(0), -1)
        x = x.view(x.size(0), -1)
        BCE = -torch.sum(x*torch.log(torch.clamp(recon_x, min=1e-10))+
            (1-x)*torch.log(torch.clamp(1-recon_x, min=1e-10)), 1)
        KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), 1)
        # Normalise by same number of elements as in reconstruction
        loss = torch.mean(BCE + KLD)

        return loss

    def forward(self, x):
        h = self.encoder(x)
        h = h.view(-1, self.hidden_size)
        mu = self._enc_mu(h)
        logvar = self._enc_log_sigma(h)
        z = self.reparameterize(mu, logvar)
        return self.decode(z), mu, logvar

    def fit(self, trainloader, validloader, lr=0.001, batch_size=128, num_epochs=10):
        use_cuda = torch.cuda.is_available()
        if use_cuda:
            self.cuda()

        optimizer = optim.Adam(filter(lambda p: p.requires_grad, self.parameters()), lr=lr)

        # validate
        self.eval()
        valid_loss = 0.0
        for batch_idx, (inputs, _) in enumerate(validloader):
            inputs = inputs.float()
            if use_cuda:
                inputs = inputs.cuda()
            inputs = Variable(inputs)
            outputs, mu, logvar = self.forward(inputs)

            loss = self.loss_function(outputs, inputs, mu, logvar)
            valid_loss += loss.data[0]*len(inputs)
            # total_loss += valid_recon_loss.data[0] * inputs.size()[0]
            # total_num += inputs.size()[0]

        # valid_loss = total_loss / total_num
        print("#Epoch -1: Valid Loss: %.5f" % (valid_loss / len(validloader.dataset)))

        for epoch in range(num_epochs):
            # train 1 epoch
            self.train()
            train_loss = 0
            for batch_idx, (inputs, _) in enumerate(trainloader):
                inputs = inputs.float()
                if use_cuda:
                    inputs = inputs.cuda()
                optimizer.zero_grad()
                inputs = Variable(inputs)
                
                outputs, mu, logvar = self.forward(inputs)
                loss = self.loss_function(outputs, inputs, mu, logvar)
                train_loss += loss.data[0]*len(inputs)
                loss.backward()
                optimizer.step()
                # print("    #Iter %3d: Reconstruct Loss: %.3f" % (
                #     batch_idx, recon_loss.data[0]))

            # validate
            self.eval()
            valid_loss = 0.0
            for batch_idx, (inputs, _) in enumerate(validloader):
                inputs = inputs.float()
                if use_cuda:
                    inputs = inputs.cuda()
                inputs = Variable(inputs)
                outputs, mu, logvar = self.forward(inputs)

                loss = self.loss_function(outputs, inputs, mu, logvar)
                valid_loss += loss.data[0]*len(inputs)
                # total_loss += valid_recon_loss.data[0] * inputs.size()[0]
                # total_num += inputs.size()[0]

                # view reconstruct
                if batch_idx == 0:
                    n = min(inputs.size(0), 8)
                    comparison = torch.cat([inputs.view(-1, self.nChannels, self.width, self.height)[:n],
                                            outputs.view(-1, self.nChannels, self.width, self.height)[:n]])
                    save_image(comparison.data.cpu(),
                                 'results/vae/reconstruct/reconstruction_' + str(epoch) + '.png', nrow=n)

            # valid_loss = total_loss / total_num
            print("#Epoch %3d: Train Loss: %.5f, Valid Loss: %.5f" % (
                epoch, train_loss / len(trainloader.dataset), valid_loss / len(validloader.dataset)))

            sample = Variable(torch.randn(64, self.z_dim))
            if use_cuda:
               sample = sample.cuda()
            sample = self.decode(sample).cpu()
            save_image(sample.data.view(64, self.nChannels, self.width, self.height),
                       'results/vae/sample_' + str(epoch) + '.png')
