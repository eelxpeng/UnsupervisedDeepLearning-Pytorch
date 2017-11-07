import torch
import torch.nn as nn
from torch.nn import Parameter
import torch.nn.functional as F
import torch.optim as optim
import torchvision
from torchvision import datasets, transforms
from torch.autograd import Variable

import numpy as np
import math
from udlp.utils import Dataset, masking_noise
from udlp.ops import MSELoss, BCELoss

class DenoisingAutoencoder(nn.Module):
    def __init__(self, in_features, out_features, activation="relu"):
        super(self.__class__, self).__init__()
        self.weight = Parameter(torch.Tensor(out_features, in_features))
        self.bias = Parameter(torch.Tensor(out_features))
        self.vbias = Parameter(torch.Tensor(in_features))
        
        if activation=="relu":
            self.enc_act_func = nn.ReLU()
        elif activation=="sigmoid":
            self.enc_act_func = nn.Sigmoid()
        self.dropout = nn.Dropout(p=0.2)

        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        self.bias.data.uniform_(-stdv, stdv)
        stdv = 1. / math.sqrt(self.vbias.size(0))
        self.vbias.data.uniform_(-stdv, stdv)

    def forward(self, x):
        return self.dropout(self.enc_act_func(F.linear(x, self.weight, self.bias)))

    def encode(self, x, train=True):
        if train:
            self.dropout.train()
        else:
            self.dropout.eval()
        return self.dropout(self.enc_act_func(F.linear(x, self.weight, self.bias)))

    def decode(self, x, binary=False):
        if not binary:
            return F.linear(x, self.weight.t(), self.vbias)
        else:
            return F.sigmoid(F.linear(x, self.weight.t(), self.vbias))

    def fit(self, data_x, valid_x, lr=0.001, batch_size=128, num_epochs=10, corrupt=0.5,
        loss_type="mse"):
        """
        data_x: FloatTensor
        valid_x: FloatTensor
        """
        use_cuda = torch.cuda.is_available()
        if use_cuda:
            self.cuda()
        print("=====Denoising Autoencoding layer=======")
        optimizer = optim.Adam(filter(lambda p: p.requires_grad, self.parameters()), lr=lr, betas=(0.9, 0.9))
        if loss_type=="mse":
            criterion = MSELoss()
        elif loss_type=="cross-entropy":
            criterion = BCELoss()
        trainset = Dataset(data_x, data_x)
        trainloader = torch.utils.data.DataLoader(
            trainset, batch_size=batch_size, shuffle=True, num_workers=2)
        validset = Dataset(valid_x, valid_x)
        validloader = torch.utils.data.DataLoader(
            validset, batch_size=1000, shuffle=False, num_workers=2)

        # validate
        total_loss = 0.0
        total_num = 0
        for batch_idx, (inputs, _) in enumerate(validloader):
            inputs = inputs.view(inputs.size(0), -1).float()
            if use_cuda:
                inputs = inputs.cuda()
            inputs = Variable(inputs)
            hidden = self.encode(inputs)
            if loss_type=="cross-entropy":
                outputs = self.decode(hidden, binary=True)
            else:
                outputs = self.decode(hidden)

            valid_recon_loss = criterion(outputs, inputs)
            total_loss += valid_recon_loss.data[0] * inputs.size()[0]
            total_num += inputs.size()[0]

        valid_loss = total_loss / total_num
        print("#Epoch 0: Valid Reconstruct Loss: %.3f" % (valid_loss))

        for epoch in range(num_epochs):
            # train 1 epoch
            for batch_idx, (inputs, _) in enumerate(trainloader):
                inputs = inputs.view(inputs.size(0), -1).float()
                inputs_corr = masking_noise(inputs, corrupt)
                if use_cuda:
                    inputs = inputs.cuda()
                    inputs_corr = inputs_corr.cuda()
                optimizer.zero_grad()
                inputs = Variable(inputs)
                inputs_corr = Variable(inputs_corr)

                hidden = self.encode(inputs_corr)
                if loss_type=="cross-entropy":
                    outputs = self.decode(hidden, binary=True)
                else:
                    outputs = self.decode(hidden)
                recon_loss = criterion(outputs, inputs)
                recon_loss.backward()
                optimizer.step()

            # validate
            total_loss = 0.0
            total_num = 0
            for batch_idx, (inputs, _) in enumerate(validloader):
                inputs = inputs.view(inputs.size(0), -1).float()
                if use_cuda:
                    inputs = inputs.cuda()
                inputs = Variable(inputs)
                hidden = self.encode(inputs, train=False)
                if loss_type=="cross-entropy":
                    outputs = self.decode(hidden, binary=True)
                else:
                    outputs = self.decode(hidden)

                valid_recon_loss = criterion(outputs, inputs)
                total_loss += valid_recon_loss.data[0] * inputs.size()[0]
                total_num += inputs.size()[0]

            valid_loss = total_loss / total_num
            print("#Epoch %3d: Reconstruct Loss: %.3f, Valid Reconstruct Loss: %.3f" % (
                epoch+1, recon_loss.data[0], valid_loss))

