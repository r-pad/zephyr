import torch
import torch.nn as nn
import torch.nn.functional as F
from time import time
import numpy as np

from .pointnet import maskMax
from .base import BaseModel

from zephyr.utils.timer import TorchTimer

# import torch_cluster

class PointNet2SSG(BaseModel):
    def __init__(self, input_dim,
                 args, num_class = 1
                 ):
        super(PointNet2SSG, self).__init__(args)

        in_channel = input_dim
        extra_bottleneck_dim = args.extra_bottleneck_dim

        self.mask_channel = [] if args.mask_channel is None else args.mask_channel
        self.xyz_channel = args.xyz_channel
        self.extra_bottleneck_dim = extra_bottleneck_dim
        self.no_coord = args.no_coord
        print("PointNet2: extra_bottleneck_dim =", self.extra_bottleneck_dim)

        self.points_channel = [i for i in range(input_dim) if (i not in self.mask_channel) and (i not in self.xyz_channel)]

        print("mask:", self.mask_channel, "xyz:", self.xyz_channel, "points:", self.points_channel)
        self.npoint_list = [128, 16, None]
        self.radius_list = [0.2, 0.5, None]
        self.nsample_list = [32, 16, None]

        # self.mlp1 = [64, 64, 128]
        # self.mlp2 = [128, 128, 256]
        # self.mlp3 = [256, 512, 1024]
        # self.mlp4 = [512, 256]

        self.mlp1 = [16, 32]
        self.mlp2 = [32, 64]
        self.mlp3 = [64, 128]
        self.mlp4 = [64, 16]

        if self.no_coord:
            in_channel1 = len(self.points_channel)
            in_channel2 = self.mlp1[-1]
            in_channel3 = self.mlp2[-1]
        else:
            in_channel1 = len(self.points_channel) + len(self.xyz_channel)
            in_channel2 = self.mlp1[-1] + len(self.xyz_channel)
            in_channel3 = self.mlp2[-1] + len(self.xyz_channel)

        self.sa1 = PointNetSetAbstraction(
            npoint=self.npoint_list[0], radius=self.radius_list[0], nsample=self.nsample_list[0],
            in_channel=in_channel1,
            mlp=self.mlp1, group_all=False, no_coord = self.no_coord
        )
        self.sa2 = PointNetSetAbstraction(
            npoint=self.npoint_list[1], radius=self.radius_list[1], nsample=self.nsample_list[1],
            in_channel=in_channel2,
            mlp=self.mlp2, group_all=False, no_coord = self.no_coord
        )
        self.sa3 = PointNetSetAbstraction(
            npoint=self.npoint_list[2], radius=self.radius_list[2], nsample=self.nsample_list[2],
            in_channel=in_channel3,
            mlp=self.mlp3, group_all=True, no_coord = self.no_coord
        )
        self.fc1 = nn.Linear(self.mlp3[-1] + extra_bottleneck_dim, self.mlp4[0])
        self.bn1 = nn.BatchNorm1d(self.mlp4[0])
        self.drop1 = nn.Dropout(0.4)
        self.fc2 = nn.Linear(self.mlp4[0], self.mlp4[1])
        self.bn2 = nn.BatchNorm1d(self.mlp4[1])
        self.drop2 = nn.Dropout(0.4)
        self.fc3 = nn.Linear(self.mlp4[1], num_class)

    def forward(self, data):
        x = data['point_x']
        # print(x.shape)

        if len(x.shape) >= 4:
            x = x.squeeze()
        x = x.transpose(1, 2) # (batch_size, n_features, n_points)

        xyz = x[:, self.xyz_channel, :]
        mask = x[:, self.mask_channel, :].sum(dim=1, keepdim=True) >= len(self.mask_channel)
        points = x[:, self.points_channel, :]

        # print(mask.shape)
        B, _, _ = xyz.shape

        l1_xyz, l1_points, l1_mask = self.sa1(xyz, points, mask)
        l2_xyz, l2_points, l2_mask = self.sa2(l1_xyz, l1_points, l1_mask)
        l3_xyz, l3_points, l3_mask = self.sa3(l2_xyz, l2_points, l2_mask)
        x = l3_points.view(B, self.mlp3[-1])

        if self.extra_bottleneck_dim > 0:
            agg_x = data['agg_x']
            x = torch.cat([x, agg_x], dim=1)

        x = self.drop1(F.relu(self.bn1(self.fc1(x))))
        x = self.drop2(F.relu(self.bn2(self.fc2(x))))
        x = self.fc3(x)
        # x = F.log_softmax(x, -1)

        return x

    def getRegLoss(self):
        return 0

    def loadPretrainedFeat(self, path, device):
        print("PointNet++: loading pretrained PointSet Absration from", path)
        pretrained_dict = torch.load(path, map_location=device)
        pretrained_dict = {k: v for k, v in pretrained_dict.items() if 'sa' in k}
        print(pretrained_dict.keys())
        model_dict = self.state_dict()
        model_dict.update(pretrained_dict)
        self.load_state_dict(model_dict)

    def freezeFeat(self):
        self.sa1.requires_grad = False
        self.sa2.requires_grad = False
        self.sa3.requires_grad = False

    def unfreezeFeat(self):
        self.sa1.requires_grad = True
        self.sa2.requires_grad = True
        self.sa3.requires_grad = True


def timeit(tag, t):
    print("{}: {}s".format(tag, time() - t))
    return time()

def pc_normalize(pc):
    l = pc.shape[0]
    centroid = np.mean(pc, axis=0)
    pc = pc - centroid
    m = np.max(np.sqrt(np.sum(pc**2, axis=1)))
    pc = pc / m
    return pc

def square_distance(src, dst):
    """
    Calculate Euclid distance between each two points.

    src^T * dst = xn * xm + yn * ym + zn * zm；
    sum(src^2, dim=-1) = xn*xn + yn*yn + zn*zn;
    sum(dst^2, dim=-1) = xm*xm + ym*ym + zm*zm;
    dist = (xn - xm)^2 + (yn - ym)^2 + (zn - zm)^2
         = sum(src**2,dim=-1)+sum(dst**2,dim=-1)-2*src^T*dst

    Input:
        src: source points, [B, N, C]
        dst: target points, [B, M, C]
    Output:
        dist: per-point square distance, [B, N, M]
    """
    B, N, _ = src.shape
    _, M, _ = dst.shape
    dist = -2 * torch.matmul(src, dst.permute(0, 2, 1))
    dist += torch.sum(src ** 2, -1).view(B, N, 1)
    dist += torch.sum(dst ** 2, -1).view(B, 1, M)
    return dist


def index_points(points, idx):
    """

    Input:
        points: input points data, [B, N, C]
        idx: sample index data, [B, S]
    Return:
        new_points:, indexed points data, [B, S, C]
    """
    device = points.device
    B = points.shape[0]
    view_shape = list(idx.shape)
    view_shape[1:] = [1] * (len(view_shape) - 1)
    repeat_shape = list(idx.shape)
    repeat_shape[0] = 1
    batch_indices = torch.arange(B, dtype=torch.long).to(device).view(view_shape).repeat(repeat_shape)
    new_points = points[batch_indices, idx, :]
    return new_points


def farthest_point_sample(xyz, npoint):
    """
    Input:
        xyz: pointcloud data, [B, N, 3]
        npoint: number of samples
    Return:
        centroids: sampled pointcloud index, [B, npoint]
    """
    device = xyz.device
    B, N, C = xyz.shape
    if type(B) is torch.Tensor:
        B = int(B.detach().item())
        N = int(N.detach().item())
        C = int(C.detach().item())
    centroids = torch.zeros(B, npoint, dtype=torch.long).to(device)
    distance = torch.ones(B, N).to(device) * 1e10
    farthest = torch.randint(0, N, (B,), dtype=torch.long).to(device)
    batch_indices = torch.arange(B, dtype=torch.long).to(device)
    for i in range(npoint):
        centroids[:, i] = farthest
        centroid = xyz[batch_indices, farthest, :].view(B, 1, C)
        dist = torch.sum((xyz - centroid) ** 2, -1)
        mask = dist < distance
        distance[mask] = dist[mask]
        farthest = torch.max(distance, -1)[1]
    return centroids

# def farthest_point_sample(xyz, npoint):
#     """
#     Input:
#         xyz: pointcloud data, [B, N, 3]
#         npoint: number of samples
#     Return:
#         centroids: sampled pointcloud index, [B, npoint]
#     """
#     device = xyz.device
#     B, N, C = xyz.shape
#     batch_idx = torch.arange(B, device=device).repeat_interleave(N)
#     centroid_idx = torch_cluster.fps(xyz.reshape((-1, C)), batch_idx, float(npoint)/N)
#     centroids = centroid_idx - torch.arange(B, device=device).repeat_interleave(npoint) * N
#     centroids = centroids.reshape((B, npoint))
#     return centroids

def query_ball_point(radius, nsample, xyz, new_xyz):
    """
    Input:
        radius: local region radius
        nsample: max sample number in local region
        xyz: all points, [B, N, C]
        new_xyz: query points, [B, S, C]
    Return:
        group_idx: grouped points index, [B, S, nsample]
    """
    device = xyz.device
    B, N, C = xyz.shape
    _, S, _ = new_xyz.shape
    group_idx = torch.arange(N, dtype=torch.long).to(device).view(1, 1, N).repeat([B, S, 1])
    sqrdists = square_distance(new_xyz, xyz)
    # group_idx[sqrdists > radius ** 2] = N
    # group_idx = group_idx.sort(dim=-1)[0][:, :, :nsample]
    group_dists, group_idx = sqrdists.topk(nsample, dim=-1, largest=False, sorted=True)
    mask = group_dists > radius ** 2
    group_first = group_idx[:, :, 0].view(B, S, 1).repeat([1, 1, nsample])
    group_idx[mask] = group_first[mask]
    return group_idx

def sample_and_group(npoint, radius, nsample, xyz, points, mask=None, returnfps=False):
    """
    Input:
        npoint:
        radius:
        nsample:
        xyz: input points position data, [B, N, 3]
        points: input points data, [B, N, D]
    Return:
        new_xyz: sampled points position data, [B, npoint, nsample, 3]
        new_points: sampled points data, [B, npoint, nsample, 3+D]
    """
    B, N, C = xyz.shape
    S = npoint
    fps_idx = farthest_point_sample(xyz, npoint) # [B, npoint, C]
    torch.cuda.empty_cache()
    new_xyz = index_points(xyz, fps_idx)
    torch.cuda.empty_cache()
    idx = query_ball_point(radius, nsample, xyz, new_xyz)
    torch.cuda.empty_cache()
    grouped_xyz = index_points(xyz, idx) # [B, npoint, nsample, C]

    torch.cuda.empty_cache()
    grouped_xyz_norm = grouped_xyz - new_xyz.view(B, S, 1, C)
    torch.cuda.empty_cache()

    if points is not None:
        grouped_points = index_points(points, idx)
        new_points = torch.cat([grouped_xyz_norm, grouped_points], dim=-1) # [B, npoint, nsample, C+D]
    else:
        new_points = grouped_xyz_norm

    if mask is not None:
        new_mask = index_points(mask, idx) # [B, npoint, nsample, 1]
    else:
        new_mask = None

    if returnfps:
        return new_xyz, new_points, new_mask, grouped_xyz, fps_idx
    else:
        return new_xyz, new_points, new_mask

def sample_and_group_all(xyz, points):
    """
    Input:
        xyz: input points position data, [B, N, 3]
        points: input points data, [B, N, D]
    Return:
        new_xyz: sampled points position data, [B, 1, 3]
        new_points: sampled points data, [B, 1, N, 3+D]
    """
    device = xyz.device
    B, N, C = xyz.shape
    new_xyz = torch.zeros(B, 1, C).to(device)
    grouped_xyz = xyz.view(B, 1, N, C)
    if points is not None:
        new_points = torch.cat([grouped_xyz, points.view(B, 1, N, -1)], dim=-1)
    else:
        new_points = grouped_xyz
    return new_xyz, new_points

class PointNetSetAbstraction(nn.Module):
    def __init__(self, npoint, radius, nsample, in_channel, mlp, group_all, no_coord):
        super(PointNetSetAbstraction, self).__init__()
        self.npoint = npoint
        self.radius = radius
        self.nsample = nsample
        self.mlp_convs = nn.ModuleList()
        self.mlp_bns = nn.ModuleList()
        self.no_coord = no_coord
        last_channel = in_channel
        for out_channel in mlp:
            self.mlp_convs.append(nn.Conv2d(last_channel, out_channel, 1))
            self.mlp_bns.append(nn.BatchNorm2d(out_channel))
            last_channel = out_channel
        self.group_all = group_all

    def forward(self, xyz, points, mask = None):
        """
        Input:
            xyz: input points position data, [B, C, N]
            points: input points data, [B, D, N]
            N: number of input points
        Return:
            new_xyz: sampled points position data, [B, C, S]
            new_points_concat: sample points feature data, [B, D', S]
            S: (n_points) number of output points
        """
        xyz = xyz.permute(0, 2, 1)
        if points is not None:
            points = points.permute(0, 2, 1)
        if mask is not None:
            mask = mask.permute(0, 2, 1)

        if self.group_all:
            new_xyz, new_points = sample_and_group_all(xyz, points)
            # new_mask = mask.permute(0, 2, 1)
            new_mask = mask.unsqueeze(1)
        else:
            new_xyz, new_points, new_mask = sample_and_group(self.npoint, self.radius, self.nsample, xyz, points, mask)

        if self.no_coord:
            new_points = new_points[:, :, :, new_xyz.shape[-1]: ]

        # new_xyz: sampled points position data, [B, npoint, C]
        # new_mask: sampled mask data: [B, npoint, nsample, 1]
        # new_points: sampled points data, [B, npoint, nsample, C+D]

        new_points = new_points.permute(0, 3, 2, 1) # [B, C+D, nsample,npoint]
        for i, conv in enumerate(self.mlp_convs):
            bn = self.mlp_bns[i]
            new_points =  F.relu(bn(conv(new_points)))

        if new_mask is not None:
            new_mask = new_mask.permute(0, 3, 2, 1) # [B, 1, nsample, npoint]
            new_points = maskMax(new_points, new_mask, dim=2, keepdim=False) # [B, C+D, npoint]
            new_mask = new_mask.sum(2) > 0 # (B, 1, npoint)
            new_xyz = new_xyz.permute(0, 2, 1) # (B, C, npoint)
            return new_xyz, new_points, new_mask
        else:
            new_points = torch.max(new_points, 2)[0]
            new_xyz = new_xyz.permute(0, 2, 1)
            return new_xyz, new_points

class PointNetSetAbstractionMsg(nn.Module):
    def __init__(self, npoint, radius_list, nsample_list, in_channel, mlp_list):
        super(PointNetSetAbstractionMsg, self).__init__()
        self.npoint = npoint
        self.radius_list = radius_list
        self.nsample_list = nsample_list
        self.conv_blocks = nn.ModuleList()
        self.bn_blocks = nn.ModuleList()
        for i in range(len(mlp_list)):
            convs = nn.ModuleList()
            bns = nn.ModuleList()
            last_channel = in_channel + 3
            for out_channel in mlp_list[i]:
                convs.append(nn.Conv2d(last_channel, out_channel, 1))
                bns.append(nn.BatchNorm2d(out_channel))
                last_channel = out_channel
            self.conv_blocks.append(convs)
            self.bn_blocks.append(bns)

    def forward(self, xyz, points):
        """
        Input:
            xyz: input points position data, [B, C, N]
            points: input points data, [B, D, N]
        Return:
            new_xyz: sampled points position data, [B, C, S]
            new_points_concat: sample points feature data, [B, D', S]
        """
        xyz = xyz.permute(0, 2, 1)
        if points is not None:
            points = points.permute(0, 2, 1)

        B, N, C = xyz.shape
        S = self.npoint
        new_xyz = index_points(xyz, farthest_point_sample(xyz, S))
        new_points_list = []
        for i, radius in enumerate(self.radius_list):
            K = self.nsample_list[i]
            group_idx = query_ball_point(radius, K, xyz, new_xyz)
            grouped_xyz = index_points(xyz, group_idx)
            grouped_xyz -= new_xyz.view(B, S, 1, C)
            if points is not None:
                grouped_points = index_points(points, group_idx)
                grouped_points = torch.cat([grouped_points, grouped_xyz], dim=-1)
            else:
                grouped_points = grouped_xyz

            grouped_points = grouped_points.permute(0, 3, 2, 1)  # [B, D, K, S]
            for j in range(len(self.conv_blocks[i])):
                conv = self.conv_blocks[i][j]
                bn = self.bn_blocks[i][j]
                grouped_points =  F.relu(bn(conv(grouped_points)))
            new_points = torch.max(grouped_points, 2)[0]  # [B, D', S]
            new_points_list.append(new_points)

        new_xyz = new_xyz.permute(0, 2, 1)
        new_points_concat = torch.cat(new_points_list, dim=1)
        return new_xyz, new_points_concat


class PointNetFeaturePropagation(nn.Module):
    def __init__(self, in_channel, mlp):
        super(PointNetFeaturePropagation, self).__init__()
        self.mlp_convs = nn.ModuleList()
        self.mlp_bns = nn.ModuleList()
        last_channel = in_channel
        for out_channel in mlp:
            self.mlp_convs.append(nn.Conv1d(last_channel, out_channel, 1))
            self.mlp_bns.append(nn.BatchNorm1d(out_channel))
            last_channel = out_channel

    def forward(self, xyz1, xyz2, points1, points2):
        """
        Input:
            xyz1: input points position data, [B, C, N]
            xyz2: sampled input points position data, [B, C, S]
            points1: input points data, [B, D, N]
            points2: input points data, [B, D, S]
        Return:
            new_points: upsampled points data, [B, D', N]
        """
        xyz1 = xyz1.permute(0, 2, 1)
        xyz2 = xyz2.permute(0, 2, 1)

        points2 = points2.permute(0, 2, 1)
        B, N, C = xyz1.shape
        _, S, _ = xyz2.shape

        if S == 1:
            interpolated_points = points2.repeat(1, N, 1)
        else:
            dists = square_distance(xyz1, xyz2)
            dists, idx = dists.sort(dim=-1)
            dists, idx = dists[:, :, :3], idx[:, :, :3]  # [B, N, 3]

            dist_recip = 1.0 / (dists + 1e-8)
            norm = torch.sum(dist_recip, dim=2, keepdim=True)
            weight = dist_recip / norm
            interpolated_points = torch.sum(index_points(points2, idx) * weight.view(B, N, 3, 1), dim=2)

        if points1 is not None:
            points1 = points1.permute(0, 2, 1)
            new_points = torch.cat([points1, interpolated_points], dim=-1)
        else:
            new_points = interpolated_points

        new_points = new_points.permute(0, 2, 1)
        for i, conv in enumerate(self.mlp_convs):
            bn = self.mlp_bns[i]
            new_points = F.relu(bn(conv(new_points)))
        return new_points
