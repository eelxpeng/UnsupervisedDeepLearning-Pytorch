import torch
import torch.nn as nn
from torch.nn import Parameter
from torch.autograd import Variable
import torch.nn.functional as F
import math


class MSELoss(nn.Module):
    def __init__(self):
        super(self.__class__, self).__init__()

    def forward(self, input, target):
        return torch.mean(torch.sum((input-target)**2, 1))

class BCELoss(nn.Module):
    def __init__(self):
        super(self.__class__, self).__init__()

    def forward(self, input, target):
        return -torch.mean(torch.sum(target*torch.log(torch.clamp(input, min=1e-10))+
            (1-target)*torch.log(torch.clamp(1-input, min=1e-10)), 1))
