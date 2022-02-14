import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .base import BaseModel

class MLP(BaseModel):
    def __init__(self, input_dim, args):
        super(MLP, self).__init__(args)
        self.fc1 = nn.Linear(input_dim, 64)
        self.fc2 = nn.Linear(64, 64)
        self.fc3 = nn.Linear(64, 1)

        self.reg_weight = args.reg_weight
        self.reg_type = args.reg_type

    def forward(self, data):
        x = data['agg_x'].squeeze()
        x = F.leaky_relu(self.fc1(x))
        x = F.leaky_relu(self.fc2(x))
        x = self.fc3(x)
        return x

    def getRegLoss(self):
        reg_type = self.reg_type
        loss = 0
        if reg_type == 'l2':
            loss += (self.fc1.weight**2).sum()
            loss += (self.fc2.weight**2).sum()
            loss += (self.fc3.weight**2).sum()
        elif args.reg_type == 'gl':
            group_lasso_loss = self.fc1.weight.norm(p=2, dim=0)
            loss += group_lasso_loss.sum()
        else:
            raise Exception("MLP: Unknown reg_type:", args.reg_type)
        loss *= self.reg_weight
        return loss

class LogReg(BaseModel):
    def __init__(self, input_dim, args):
        super(LogReg, self).__init__(args)
        self.fc1 = nn.Linear(input_dim, 1)
        self.reg_type = args.reg_type
        self.reg_weight = args.reg_weight

    def forward(self, data):
        x = data['agg_x'].squeeze()
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
            raise Exception("LogReg: Unknown reg_type:", args.reg_type)
        loss *= self.reg_weight
        return loss
