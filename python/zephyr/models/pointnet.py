from __future__ import print_function
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

class PointNet(BaseModel):
    def __init__(self, input_dim, args):
        super(PointNet, self).__init__(args)

        mask_channel = [] if args.mask_channel is None else args.mask_channel
        pn_pool = args.pn_pool
        bottleneck_dim = args.bottleneck_dim
        extra_bottleneck_dim = args.extra_bottleneck_dim

        print("input_dim:", input_dim, "mask_channel:", mask_channel, "extra_bottleneck_dim:", extra_bottleneck_dim)
        self.network = PointNetCls(
            input_dim = input_dim, k = 1,
            feature_transform = False, input_transform = False,
            mask_channel = mask_channel, pool = pn_pool,
            bottleneck_dim = bottleneck_dim, extra_bottleneck_dim = extra_bottleneck_dim,
            )

    def forward(self, data):
        x = data['point_x'].squeeze()
        if 'agg_x' in data.keys():
            agg_x = data['agg_x'].squeeze()
        else:
            agg_x = None

        if len(x.shape) == 2:
            x = x.unsqueeze(0)
        if len(x.shape) >= 4:
            x = x.squeeze()
        x = x.transpose(1, 2)

        # x should be of shape (batch_size, n_features, n_points)
        score, _, _ = self.network(x, agg_x)
        return score

    def getRegLoss(self):
        return 0

    def loadPretrainedFeat(self, path):
        print("PointNet: loading pretrained PN feat from", path)
        pretrained_dict = torch.load(path)
        pretrained_dict = {k: v for k, v in pretrained_dict.items() if 'feat' in k}
        model_dict = self.state_dict()
        model_dict.update(pretrained_dict)
        self.load_state_dict(model_dict)

    def freezeFeat(self):
        self.network.feat.requires_grad = False

    def unfreezeFeat(self):
        self.network.feat.requires_grad = True

class STN3d(nn.Module):
    def __init__(self, input_dim=3):
        super(STN3d, self).__init__()
        self.conv1 = torch.nn.Conv1d(input_dim, 64, 1)
        self.conv2 = torch.nn.Conv1d(64, 128, 1)
        self.conv3 = torch.nn.Conv1d(128, 1024, 1)
        self.fc1 = nn.Linear(1024, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, 9)
        self.relu = nn.ReLU()

        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(128)
        self.bn3 = nn.BatchNorm1d(1024)
        self.bn4 = nn.BatchNorm1d(512)
        self.bn5 = nn.BatchNorm1d(256)

    def forward(self, x):
        batchsize = x.size()[0]
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        x = torch.max(x, 2, keepdim=True)[0]
        x = x.view(-1, 1024)

        x = F.relu(self.bn4(self.fc1(x)))
        x = F.relu(self.bn5(self.fc2(x)))
        x = self.fc3(x)

        iden = Variable(torch.from_numpy(np.array([1,0,0,0,1,0,0,0,1]).astype(np.float32))).view(1,9).repeat(batchsize,1)
        if x.is_cuda:
            iden = iden.to(x.device)
        x = x + iden
        x = x.view(-1, 3, 3)
        return x

class STNkd(nn.Module):
    def __init__(self, k=64):
        super(STNkd, self).__init__()
        self.conv1 = torch.nn.Conv1d(k, 64, 1)
        self.conv2 = torch.nn.Conv1d(64, 128, 1)
        self.conv3 = torch.nn.Conv1d(128, 1024, 1)
        self.fc1 = nn.Linear(1024, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, k*k)
        self.relu = nn.ReLU()

        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(128)
        self.bn3 = nn.BatchNorm1d(1024)
        self.bn4 = nn.BatchNorm1d(512)
        self.bn5 = nn.BatchNorm1d(256)

        self.k = k

    def forward(self, x):
        batchsize = x.size()[0]
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        x = torch.max(x, 2, keepdim=True)[0]
        x = x.view(-1, 1024)

        x = F.relu(self.bn4(self.fc1(x)))
        x = F.relu(self.bn5(self.fc2(x)))
        x = self.fc3(x)

        iden = Variable(torch.from_numpy(np.eye(self.k).flatten().astype(np.float32))).view(1,self.k*self.k).repeat(batchsize,1)
        if x.is_cuda:
            iden = iden.to(x.device)
        x = x + iden
        x = x.view(-1, self.k, self.k)
        return x

def maskMax(x, mask, dim, keepdim):
    if not mask is None:
        mask = mask.float()
        x = x - (1-mask) * 1e7
    pooled = x.max(dim=dim, keepdim=keepdim)[0]
    return pooled

def maskAvg(x, mask, dim, keepdim):
    if not mask is None:
        mask = mask.float()
        x = x * mask
        pooled = x.sum(dim=dim, keepdim=keepdim) / mask.sum(dim=dim, keepdim=keepdim)
    else:
        pooled = x.mean(dim=dim, keepdim=keepdim)
    return pooled

def maskCat(x, mask, dim, keepdim):
    max_pooled = maskMax(x, mask, dim, keepdim)
    avg_pooled = maskAvg(x, mask, dim, keepdim)
    pooled = torch.cat([max_pooled, avg_pooled], axis=1)
    return pooled

class PointNetfeat(nn.Module):
    def __init__(self,
                 input_dim=3, global_feat = True,
                 feature_transform = False, input_transform = False,
                 mask_channel = None, pool = "max",
                 bottleneck_dim = 1024,
                 ):
        super(PointNetfeat, self).__init__()
        input_dim -= len(mask_channel)

        if bottleneck_dim >= 1024:
            self.conv1 = torch.nn.Conv1d(input_dim, 64, 1)
            self.conv2 = torch.nn.Conv1d(64, 128, 1)
            self.conv3 = torch.nn.Conv1d(128, bottleneck_dim, 1)
            self.bn1 = nn.BatchNorm1d(64)
            self.bn2 = nn.BatchNorm1d(128)
            self.bn3 = nn.BatchNorm1d(bottleneck_dim)
        else:
            self.conv1 = torch.nn.Conv1d(input_dim, 16, 1)
            self.conv2 = torch.nn.Conv1d(16, bottleneck_dim, 1)
            self.bn1 = nn.BatchNorm1d(16)
            self.bn2 = nn.BatchNorm1d(bottleneck_dim)

        self.global_feat = global_feat
        self.feature_transform = feature_transform
        self.input_transform = input_transform
        if self.input_transform:
            print("Using input_transform")
            self.stn = STNkd(k=input_dim)
        if self.feature_transform:
            print("Using feature_transform")
            self.fstn = STNkd(k=64)

        '''Use masked or unmasked version of the pooling layer instead of simple max pooling'''
        self.pool = pool
        self.mask_channel = mask_channel
        print("PointNetfeat: mask_channel =", mask_channel)
        if self.pool == "max":
            print("PointNetfeat: Using masked max pooling")
            self.pool_layer = maskMax
        elif self.pool == "avg":
            print("PointNetfeat: Using masked average pooling")
            self.pool_layer = maskAvg
        elif self.pool == "cat":
            print("PointNetfeat: Using masked max+avg cat pooling")
            self.pool_layer = maskCat
        else:
            raise Exception("PointNetfeat: Unknown pool layer name:", self.pool)

        '''Control the final feature dimension of the PointNet output'''
        self.bottleneck_dim = bottleneck_dim

    def forward(self, x):
        if self.mask_channel is not None and len(self.mask_channel) > 0:
            mask = x[:, self.mask_channel, :].sum(dim=1, keepdim=True) >= len(self.mask_channel)
            # mask = x[:, self.mask_channel, :].sum(dim=1, keepdim=True) > 0
            mask = mask.detach()
            unmask_indices = [_ for _ in range(x.shape[1]) if _ not in self.mask_channel]
            x = x[:, unmask_indices, :]
            if (mask.sum(-1).squeeze() == 0).sum() > 0:
                print("Error! Invalid hypotheses = ", (mask.sum(-1).squeeze() == 0).sum())
        else:
            mask = None

        n_pts = x.size()[2]

        if self.input_transform:
            trans = self.stn(x)
            x = x.transpose(2, 1)
            x = torch.bmm(x, trans)
            x = x.transpose(2, 1)
        else:
            trans = None

        x = F.relu(self.bn1(self.conv1(x)))

        if self.feature_transform:
            trans_feat = self.fstn(x)
            x = x.transpose(2,1)
            x = torch.bmm(x, trans_feat)
            x = x.transpose(2,1)
        else:
            trans_feat = None

        pointfeat = x

        x = F.relu(self.bn2(self.conv2(x)))

        if self.bottleneck_dim >= 1024:
            x = self.bn3(self.conv3(x))

        if mask is not None:
            mask = mask.expand(*x.shape)
        x = self.pool_layer(x, mask, dim=2, keepdim=True)

        x = x.squeeze()
        if self.global_feat:
            return x, trans, trans_feat
        else:
            x = x.unsqueeze(-1).repeat(1, 1, n_pts)
            return torch.cat([x, pointfeat], 1), trans, trans_feat

class PointNetCls(nn.Module):
    def __init__(self,
                 input_dim = 3, k = 2,
                 feature_transform = False, input_transform = False,
                 mask_channel = None, pool = "max",
                 bottleneck_dim = 1024, extra_bottleneck_dim = 0,
                 ):
        super(PointNetCls, self).__init__()
        self.feature_transform = feature_transform
        self.bottleneck_dim = bottleneck_dim
        self.feat = PointNetfeat(
            input_dim=input_dim, global_feat=True,
            feature_transform = feature_transform, input_transform = input_transform,
            mask_channel = mask_channel, pool = pool,
            bottleneck_dim = self.bottleneck_dim
            )

        # If concate the max and avg pooling
        if pool == 'cat':
            self.fc_in_dim = 2 * self.bottleneck_dim
        else:
            self.fc_in_dim = self.bottleneck_dim

        # If extra input from the aggregated scoring features
        self.fc_in_dim += extra_bottleneck_dim


        self.relu = nn.ReLU()
        if self.bottleneck_dim == 1024:
            print("PointNetCls: Useing feature dim 1024")
            self.fc1 = nn.Linear(1024, 512)
            self.fc2 = nn.Linear(512, 256)
            self.fc3 = nn.Linear(256, k)
            self.dropout = nn.Dropout(p=0.3)
            self.bn1 = nn.BatchNorm1d(512)
            self.bn2 = nn.BatchNorm1d(256)

            self.network = nn.Sequential(
                self.fc1, self.bn1, self.relu,
                self.fc2, self.dropout, self.bn2, self.relu,
                self.fc3
            )
        elif self.bottleneck_dim <= 128:
            # self.fc1 = nn.Linear(128, 64)
            # self.fc2 = nn.Linear(64, k)
            # self.bn1 = nn.BatchNorm1d(64)
            # self.dropout = nn.Dropout(p=0.3)
            #
            # self.network = nn.Sequential(
            #     self.fc1, self.dropout, self.bn1, self.relu,
            #     self.fc2
            # )
            self.fc1_dim = 64
            self.fc1 = nn.Linear(self.fc_in_dim, self.fc1_dim)
            self.bn1 = nn.BatchNorm1d(self.fc1_dim)
            self.fc2 = nn.Linear(self.fc1_dim, k)
            self.dropout = nn.Dropout(p=0.3)

            self.network = nn.Sequential(
                self.fc1, self.bn1, self.relu,
                self.fc2
            )
        else:
            raise Exception("Unimplemented bottleneck_dim:", self.bottleneck_dim)


    def forward(self, x, score_x = None):
        x, trans, trans_feat = self.feat(x)

        if not score_x is None:
            score_x = score_x.squeeze()
            x = torch.cat([x, score_x], -1)
        x = self.network(x)

        # return F.log_softmax(x, dim=1), trans, trans_feat
        return x, trans, trans_feat

class PointNetDenseCls(nn.Module):
    def __init__(self, input_dim=3, k=2, feature_transform=False):
        super(PointNetDenseCls, self).__init__()
        self.k = k
        self.feature_transform=feature_transform
        self.feat = PointNetfeat(input_dim=input_dim, global_feat=False, feature_transform=feature_transform)
        self.conv1 = torch.nn.Conv1d(1088, 512, 1)
        self.conv2 = torch.nn.Conv1d(512, 256, 1)
        self.conv3 = torch.nn.Conv1d(256, 128, 1)
        self.conv4 = torch.nn.Conv1d(128, self.k, 1)
        self.bn1 = nn.BatchNorm1d(512)
        self.bn2 = nn.BatchNorm1d(256)
        self.bn3 = nn.BatchNorm1d(128)

    def forward(self, x):
        batchsize = x.size()[0]
        n_pts = x.size()[2]
        x, trans, trans_feat = self.feat(x)
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        x = self.conv4(x)
        x = x.transpose(2,1).contiguous()

        # x = F.log_softmax(x.view(-1,self.k), dim=-1)

        x = x.view(batchsize, n_pts, self.k)
        return x, trans, trans_feat

def feature_transform_regularizer(trans):
    d = trans.size()[1]
    batchsize = trans.size()[0]
    I = torch.eye(d)[None, :, :]
    if trans.is_cuda:
        I = I.cuda()
    loss = torch.mean(torch.norm(torch.bmm(trans, trans.transpose(2,1)) - I, dim=(1,2)))
    return loss

if __name__ == '__main__':
    # sim_data = Variable(torch.rand(1, 38, 1001))
    sim_data = Variable(torch.rand(32,3,2500))
    trans = STN3d()
    out = trans(sim_data)
    print('stn', out.size())
    print('loss', feature_transform_regularizer(out))

    sim_data_64d = Variable(torch.rand(32, 64, 2500))
    trans = STNkd(k=64)
    out = trans(sim_data_64d)
    print('stn64d', out.size())
    print('loss', feature_transform_regularizer(out))

    pointfeat = PointNetfeat(global_feat=True)
    out, _, _ = pointfeat(sim_data)
    print('global feat', out.size())

    pointfeat = PointNetfeat(global_feat=False)
    out, _, _ = pointfeat(sim_data)
    print('point feat', out.size())

    cls = PointNetCls(k = 5)
    out, _, _ = cls(sim_data)
    print('class', out.size())

    seg = PointNetDenseCls(k = 3)
    out, _, _ = seg(sim_data)
    print('seg', out.size())
