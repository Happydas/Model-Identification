import os
import time
import numpy as np
import scipy.io as sio
import matplotlib.pyplot as plt
import torch
import torch.nn as nn

from torch.autograd import grad
from torch.optim import Adam
from torch.utils.data import Dataset, DataLoader
from torch.optim.lr_scheduler import ExponentialLR


class PINN(nn.Module):

    def __init__(self, sizes, activation=nn.Tanh()):
        super(PINN, self).__init__()
        layer = []
        for i in range(len(sizes)-2):
            layer += [
                nn.Linear(in_features=sizes[i], out_features=sizes[i+1]),
                activation
            ]

        layer += [nn.Linear(in_features=sizes[-2], out_features=sizes[-1])]

        self.net = nn.Sequential(*layer)

    def forward(self, x):
        return self.net(x)



