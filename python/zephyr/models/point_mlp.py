import sys
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.utils.data
from torch.autograd import Variable
import numpy as np
import torch.nn.functional as F
from functools import partial

from .base import BaseModel

class PointMLP(BaseModel):
    def __init__(self, input_dim, args):
        super(PointMLP, self).__init__(args)
        self.fc1 = nn.Linear(input_dim, 1)
        self.reg_type = args.reg_type
        self.reg_weight = args.reg_weight
        self.args = args

    def forward(self, data):
        x = data['point_x'].squeeze()
        x = x.sum(1)
        x = self.fc1(x)
        return x

    def getRegLoss(self):
        reg_type = self.reg_type
        loss = 0
        if reg_type == 'l1':
            loss += self.fc1.weight.abs().sum()
        elif reg_type == 'l2':
            loss += (self.fc1.weight**2).sum()
        else:
            raise Exception("LogReg: Unknown reg_type:", self.args.reg_type)
        loss *= self.reg_weight
        return loss