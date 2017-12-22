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

def buildNetwork(layers, activation="relu", dropout=0.5):
    net = []
    for i in range(1, len(layers)):
        net.append(nn.Linear(layers[i-1], layers[i]))
        if activation=="relu":
            net.append(nn.ReLU())
        elif activation=="sigmoid":
            net.append(nn.Sigmoid())
        if dropout > 0:
            net.append(nn.Dropout(dropout))
    return nn.Sequential(*net)

def loss_function(recon_x, x, mu, logvar):
    BCE = F.binary_cross_entropy(recon_x, x)

    # see Appendix B from VAE paper:
    # Kingma and Welling. Auto-Encoding Variational Bayes. ICLR, 2014
    # https://arxiv.org/abs/1312.6114
    # 0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
    KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    # Normalise by same number of elements as in reconstruction
    KLD /= x.size()[0] * x.size()[1]

    return BCE + KLD

class VAE(nn.Module):
    def __init__(self, input_dim=784, z_dim=10, binary=True,
        encodeLayer=[400], decodeLayer=[400]):
        super(self.__class__, self).__init__()
        self.encoder = buildNetwork([input_dim] + encodeLayer)
        self.decoder = buildNetwork([z_dim] + decodeLayer)
        self._enc_mu = nn.Linear(encodeLayer[-1], z_dim)
        self._enc_log_sigma = nn.Linear(encodeLayer[-1], z_dim)
        self._dec = nn.Linear(decodeLayer[-1], input_dim)
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
        h = self.decoder(z)
        x = self._dec(h)
        if self._dec_act is not None:
            x = self._dec_act(x)
        return x

    def forward(self, x):
        h = self.encoder(x)
        mu = self._enc_mu(h)
        logvar = self._enc_log_sigma(h)
        z = self.reparameterize(mu, logvar)
        return self.decode(z), mu, logvar

    def fit(self, trainloader, validloader, lr=0.001, batch_size=128, num_epochs=10):
        use_cuda = torch.cuda.is_available()
        if use_cuda:
            self.cuda()

        optimizer = optim.Adam(filter(lambda p: p.requires_grad, self.parameters()), lr=lr)
        # optimizer = optim.SGD(filter(lambda p: p.requires_grad, self.parameters()), lr=lr, momentum=0.9)
        # criterion = nn.MSELoss(size_average=False)
        # criterion = nn.MSELoss()
        # criterion = MSELoss()
        # if loss_type=="cross-entropy":
        #     criterion = BCELoss()
        # elif loss_type=="mse":
        #     criterion = MSELoss()
        # trainset = Dataset(data_x, data_x)
        # trainloader = torch.utils.data.DataLoader(
        #     trainset, batch_size=batch_size, shuffle=True, num_workers=2)
        # validset = Dataset(valid_x, valid_x)
        # validloader = torch.utils.data.DataLoader(
        #     validset, batch_size=1000, shuffle=False, num_workers=2)

        # validate
        self.eval()
        valid_loss = 0.0
        for batch_idx, (inputs, _) in enumerate(validloader):
            inputs = inputs.view(inputs.size(0), -1).float()
            if use_cuda:
                inputs = inputs.cuda()
            inputs = Variable(inputs)
            outputs, mu, logvar = self.forward(inputs)

            loss = loss_function(outputs, inputs, mu, logvar)
            valid_loss += loss.data[0]
            # total_loss += valid_recon_loss.data[0] * inputs.size()[0]
            # total_num += inputs.size()[0]

        # valid_loss = total_loss / total_num
        print("#Epoch -1: Valid Loss: %.5f" % (valid_loss / len(validloader.dataset)))

        for epoch in range(num_epochs):
            # train 1 epoch
            self.train()
            train_loss = 0
            for batch_idx, (inputs, _) in enumerate(trainloader):
                inputs = inputs.view(inputs.size(0), -1).float()
                if use_cuda:
                    inputs = inputs.cuda()
                optimizer.zero_grad()
                inputs = Variable(inputs)
                
                outputs, mu, logvar = self.forward(inputs)
                loss = loss_function(outputs, inputs, mu, logvar)
                train_loss += loss.data[0]
                loss.backward()
                optimizer.step()
                # print("    #Iter %3d: Reconstruct Loss: %.3f" % (
                #     batch_idx, recon_loss.data[0]))

            # validate
            self.eval()
            valid_loss = 0.0
            for batch_idx, (inputs, _) in enumerate(validloader):
                inputs = inputs.view(inputs.size(0), -1).float()
                if use_cuda:
                    inputs = inputs.cuda()
                inputs = Variable(inputs)
                outputs, mu, logvar = self.forward(inputs)

                loss = loss_function(outputs, inputs, mu, logvar)
                valid_loss += loss.data[0]
                # total_loss += valid_recon_loss.data[0] * inputs.size()[0]
                # total_num += inputs.size()[0]

            # valid_loss = total_loss / total_num
            print("#Epoch %3d: Train Loss: %.5f, Valid Loss: %.5f" % (
                epoch, train_loss / len(trainloader.dataset), valid_loss / len(validloader.dataset)))

            sample = Variable(torch.randn(64, 20))
            if use_cuda:
               sample = sample.cuda()
            sample = self.decode(sample).cpu()
            save_image(sample.data.view(64, 1, 28, 28),
                       'results/vae/sample_' + str(epoch) + '.png')
