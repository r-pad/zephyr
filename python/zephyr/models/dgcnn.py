#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@Author: Yue Wang
@Contact: yuewangx@mit.edu
@File: model.py
@Time: 2018/10/13 6:35 PM

Modified by
@Author: An Tao
@Contact: ta19@mails.tsinghua.edu.cn
@Time: 2020/3/9 9:32 PM
"""

import os
import sys
import copy
import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.init as init
import torch.nn.functional as F

from .base import BaseModel

class DGCNN(BaseModel):
    def __init__(self, input_dim, args, num_class=1):
        super(DGCNN, self).__init__(args)
        self.input_dim = input_dim
        self.mask_channel = args.mask_channel
        self.xyz_channel = args.xyz_channel
        self.extra_bottleneck_dim = args.extra_bottleneck_dim
        self.net = DGCNN_cls(args, self.input_dim, output_channels=1, pos_idx=xyz_channel,
                             extra_bottleneck_dim=self.extra_bottleneck_dim)

    def forward(self, data):
        x = data['point_x']

        if self.extra_bottleneck_dim > 0:
            agg_x = data['agg_x']
        else:
            agg_x = None

        if len(x.shape) >= 4:
            x = x.squeeze()
        x = x.transpose(1, 2) # (batch_size, n_features, n_points)

        return self.net(x, agg_x)

    def getRegLoss(self):
        return 0

def knn(x, k):
    inner = -2*torch.matmul(x.transpose(2, 1), x)
    xx = torch.sum(x**2, dim=1, keepdim=True)
    pairwise_distance = -xx - inner - xx.transpose(2, 1)

    idx = pairwise_distance.topk(k=k, dim=-1)[1]   # (batch_size, num_points, k)
    return idx

def get_graph_feature(x, k=20, idx=None, pos_idx=None):
    batch_size = x.size(0)
    num_points = x.size(2)
    x = x.view(batch_size, -1, num_points)
    if idx is None:
        if pos_idx is None:
            # print("regular kNN")
            idx = knn(x, k=k)   # (batch_size, num_points, k)
        else:
            # print("indexed kNN")
            idx = knn(x[:, pos_idx], k=k)
    device = torch.device('cuda')

    idx_base = torch.arange(0, batch_size, device=device).view(-1, 1, 1)*num_points

    idx = idx + idx_base

    idx = idx.view(-1)

    _, num_dims, _ = x.size()

    x = x.transpose(2, 1).contiguous()   # (batch_size, num_points, num_dims)  -> (batch_size*num_points, num_dims) #   batch_size * num_points * k + range(0, batch_size*num_points)
    feature = x.view(batch_size*num_points, -1)[idx, :]
    feature = feature.view(batch_size, num_points, k, num_dims)
    x = x.view(batch_size, num_points, 1, num_dims).repeat(1, 1, k, 1)

    feature = torch.cat((feature-x, x), dim=3).permute(0, 3, 1, 2)

    return feature      # (batch_size, 2*num_dims, num_points, k)

class DGCNN_cls(nn.Module):
    def __init__(self, args, input_dim, output_channels=40, pos_idx=None, extra_bottleneck_dim=0):
        super(DGCNN_cls, self).__init__()
        self.args = args
        self.k = args.dgcnn_k
        self.input_dim = input_dim
        self.output_channels = output_channels
        self.pos_idx = pos_idx
        self.extra_bottleneck_dim = extra_bottleneck_dim
        # self.dim_list = [64, 64, 128, 256, args.dgcnn_emb_dims]
        # self.mlp_list = [512, 256, output_channels]
        self.dim_list = [16, 16, 16, 32, 32]
        self.mlp_list = [32, 16, output_channels]

        self.bn1 = nn.BatchNorm2d(self.dim_list[0])
        self.bn2 = nn.BatchNorm2d(self.dim_list[1])
        self.bn3 = nn.BatchNorm2d(self.dim_list[2])
        self.bn4 = nn.BatchNorm2d(self.dim_list[3])
        self.bn5 = nn.BatchNorm1d(self.dim_list[4])

        self.conv1 = nn.Sequential(nn.Conv2d(2*input_dim, self.dim_list[0], kernel_size=1, bias=False),
                                   self.bn1,
                                   nn.LeakyReLU(negative_slope=0.2))
        self.conv2 = nn.Sequential(nn.Conv2d(2*self.dim_list[0], self.dim_list[1], kernel_size=1, bias=False),
                                   self.bn2,
                                   nn.LeakyReLU(negative_slope=0.2))
        self.conv3 = nn.Sequential(nn.Conv2d(2*self.dim_list[1], self.dim_list[2], kernel_size=1, bias=False),
                                   self.bn3,
                                   nn.LeakyReLU(negative_slope=0.2))
        self.conv4 = nn.Sequential(nn.Conv2d(2*self.dim_list[2], self.dim_list[3], kernel_size=1, bias=False),
                                   self.bn4,
                                   nn.LeakyReLU(negative_slope=0.2))
        self.conv5 = nn.Sequential(nn.Conv1d(sum(self.dim_list[:4]), self.dim_list[4], kernel_size=1, bias=False),
                                   self.bn5,
                                   nn.LeakyReLU(negative_slope=0.2))

        self.linear1 = nn.Linear(2*self.dim_list[4] + extra_bottleneck_dim, self.mlp_list[0], bias=False)
        self.bn6 = nn.BatchNorm1d(self.mlp_list[0])
        self.dp1 = nn.Dropout(p=args.dgcnn_dropout)
        self.linear2 = nn.Linear(self.mlp_list[0], self.mlp_list[1])
        self.bn7 = nn.BatchNorm1d(self.mlp_list[1])
        self.dp2 = nn.Dropout(p=args.dgcnn_dropout)
        self.linear3 = nn.Linear(self.mlp_list[1], output_channels)

    def forward(self, x, agg_x = None):
        # print(x.shape)
        batch_size = x.size(0)
        torch.cuda.empty_cache()
        x = get_graph_feature(x, k=self.k, pos_idx = self.pos_idx)      # (batch_size, 3, num_points) -> (batch_size, 3*2, num_points, k)
        x = self.conv1(x)                       # (batch_size, 3*2, num_points, k) -> (batch_size, 64, num_points, k)
        x1 = x.max(dim=-1, keepdim=False)[0]    # (batch_size, 64, num_points, k) -> (batch_size, 64, num_points)

        torch.cuda.empty_cache()
        x = get_graph_feature(x1, k=self.k)     # (batch_size, 64, num_points) -> (batch_size, 64*2, num_points, k)
        x = self.conv2(x)                       # (batch_size, 64*2, num_points, k) -> (batch_size, 64, num_points, k)
        x2 = x.max(dim=-1, keepdim=False)[0]    # (batch_size, 64, num_points, k) -> (batch_size, 64, num_points)

        torch.cuda.empty_cache()
        x = get_graph_feature(x2, k=self.k)     # (batch_size, 64, num_points) -> (batch_size, 64*2, num_points, k)
        x = self.conv3(x)                       # (batch_size, 64*2, num_points, k) -> (batch_size, 128, num_points, k)
        x3 = x.max(dim=-1, keepdim=False)[0]    # (batch_size, 128, num_points, k) -> (batch_size, 128, num_points)

        torch.cuda.empty_cache()
        x = get_graph_feature(x3, k=self.k)     # (batch_size, 128, num_points) -> (batch_size, 128*2, num_points, k)
        x = self.conv4(x)                       # (batch_size, 128*2, num_points, k) -> (batch_size, 256, num_points, k)
        x4 = x.max(dim=-1, keepdim=False)[0]    # (batch_size, 256, num_points, k) -> (batch_size, 256, num_points)

        x = torch.cat((x1, x2, x3, x4), dim=1)  # (batch_size, 64+64+128+256, num_points)

        x = self.conv5(x)                       # (batch_size, 64+64+128+256, num_points) -> (batch_size, emb_dims, num_points)
        x1 = F.adaptive_max_pool1d(x, 1).view(batch_size, -1)           # (batch_size, emb_dims, num_points) -> (batch_size, emb_dims)
        x2 = F.adaptive_avg_pool1d(x, 1).view(batch_size, -1)           # (batch_size, emb_dims, num_points) -> (batch_size, emb_dims)
        x = torch.cat((x1, x2), 1)              # (batch_size, emb_dims*2)

        if self.extra_bottleneck_dim > 0:
            x = torch.cat((x, agg_x), 1)

        x = F.leaky_relu(self.bn6(self.linear1(x)), negative_slope=0.2) # (batch_size, emb_dims*2) -> (batch_size, 512)
        x = self.dp1(x)
        x = F.leaky_relu(self.bn7(self.linear2(x)), negative_slope=0.2) # (batch_size, 512) -> (batch_size, 256)
        x = self.dp2(x)
        x = self.linear3(x)                                             # (batch_size, 256) -> (batch_size, output_channels)

        return x
