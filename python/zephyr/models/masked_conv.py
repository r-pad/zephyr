import os, time

import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader
import torchvision

from oriented_features.masked.masked_resnet import masked_resnet18
from oriented_features.masked.masked_comparison_resnet import masked_comparison_resnet
from oriented_features.image_utils.distance_transform import RadialDistanceTransform
from oriented_features.image_utils.edges import generate_distance_image
from oriented_features.projective_renderer import projective_render

from pytorch_lightning.core.lightning import LightningModule

from .base import BaseModel
from oriented_features.pose_scoring_lightning.utils.image_crop import crop_resize, mask2bbox, squarePad, bbox2Square
from oriented_features.pose_scoring_lightning.utils.mask_edge import mask2edge, get_gaussian_kernel
from oriented_features.utils import TorchTimer, Timer

class ConvolutionalPoseModel(BaseModel):
    def __init__(self, args, pretrained=True):
        super(ConvolutionalPoseModel, self).__init__(args)
        self.pretrained = args.masked_pretained
        self.mask = not args.masked_no_mask
        self.input_channels = args.dim_render

        # self.crop_size = 64
        # self.pad_size = 3
        #
        # self.RGB = ("RGB" in args.dataset)
        # self.D = ("D" in args.dataset)
        # self.diff = ("diff" in args.dataset)
        # self.edge = ("edge" in args.dataset)
        # self.mask = ("mask" in args.dataset)
        # print("ConvPoseModel RGB:", self.RGB, "D:", self.D, "diff:", self.diff, "edge:", self.edge, "mask:", self.mask)
        #
        # input_channels = 0
        # if self.RGB:
        #     input_channels += 3 if self.diff else 6
        # if self.D:
        #     input_channels += 1 if self.diff else 2
        # if self.edge:
        #     input_channels += 2
        #     self.edge_blur = get_gaussian_kernel(kernel_size=11, sigma=6, channels=1)

        if not self.mask:
            self.net = torchvision.models.resnet18(pretrained=self.pretrained)
            '''Reset the input and output layer to match the actual input and output'''
            self.net.conv1 = torch.nn.Conv2d(self.input_channels, 64, kernel_size=7, stride=2, padding=3, bias=False)
            self.net.fc = torch.nn.Linear(self.net.fc.weight.shape[1], 64)
        else:
            self.net = masked_resnet18(pretrained=self.pretrained, num_classes=64, input_channels=self.input_channels)
            # self.net = masked_comparison_resnet(pretrained=True, num_classes=64, input_channels=[4,4])

        self.fc = torch.nn.Linear(64, 1)

    def forward_(self, img_obs, depth_obs, transforms, meta_data,
                model_points, model_colors, model_normals):
        with torch.no_grad():
            img_rend, depth_rend, mask_rend, ll_corner, ur_corner = projective_render(
                model_points, model_colors, model_normals,
                transforms, meta_data, gpu_id = img_obs.device, image_size = img_obs.shape[-2:], downsample=4,
                output_size = (self.crop_size-2*self.pad_size, self.crop_size-2*self.pad_size), border_size = self.pad_size
            )

            img_rend = img_rend.permute(0,3,1,2)
            depth_rend = depth_rend.unsqueeze(1)
            mask_rend = mask_rend.unsqueeze(1)

            edge_rend = mask2edge(mask_rend).float()
            edge_obs_dist = generate_distance_image(depth_obs[0,0])

            ''' The bbox is already on the obs image space
                A little formatting
            '''
            bbox_obs = [ll_corner[:, 1], ll_corner[:, 0],
                    ur_corner[:, 1], ur_corner[:, 0]]
            bbox_obs = torch.stack(bbox_obs, dim=1)
            bbox_obs = bbox_obs.reshape((bbox_obs.shape[0], 2, 2))
            bbox_obs = bbox_obs.permute((1, 2, 0))

            '''Normalize the observed edge'''
            # edge_obs_dist = edge_obs_dist.max(-1, keepdim=True)[0].max(-2, keepdim=True)[0] - edge_obs_dist
            # edge_obs_dist = 1 - edge_obs_dist

            '''Pad and crop the observed images'''
            img_obs = squarePad(img_obs)
            depth_obs = squarePad(depth_obs)
            edge_obs_dist = squarePad(edge_obs_dist)
            img_obs = crop_resize(img_obs.expand(img_rend.shape[0], -1, -1, -1), bbox_obs, self.crop_size)
            depth_obs = crop_resize(depth_obs.expand(img_rend.shape[0], -1, -1, -1), bbox_obs, self.crop_size)
            edge_obs_dist = crop_resize(edge_obs_dist.expand(img_rend.shape[0], -1, -1, -1), bbox_obs, self.crop_size)

            '''Blur and normalize the rendered edges'''
            edge_rend = self.edge_blur(edge_rend)
            edge_rend /= edge_rend.max(-1, keepdim=True)[0].max(-2, keepdim=True)[0]

            x = []

            if self.RGB:
                x += [img_obs - img_rend] if self.diff else [img_obs, img_rend]
            if self.D:
                x += [depth_obs - depth_rend] if self.diff else [depth_obs, depth_rend]
            if self.edge:
                x += [edge_obs_dist, edge_rend]

            x = torch.cat(x, dim=1)

            '''Logging the training image for debug purpose'''
            # edge_obs_dist_norm = edge_obs_dist / edge_obs_dist.max(-1, keepdim=True)[0].max(-2, keepdim=True)[0]
            # depth_obs_norm = depth_obs / depth_obs.max(-1, keepdim=True)[0].max(-2, keepdim=True)[0]
            # depth_rend_norm = depth_rend / depth_rend.max(-1, keepdim=True)[0].max(-2, keepdim=True)[0]
            # grid = torchvision.utils.make_grid([
            #     img_obs[0],
            #     depth_obs_norm[0].repeat(3, 1, 1),
            #     edge_obs_dist_norm[0].repeat(3, 1, 1),
            #     img_rend[0],
            #     depth_rend_norm[0].repeat(3, 1, 1),
            #     edge_rend[0].repeat(3, 1, 1)
            # ])
            # self.logger.experiment.add_image('training images', grid, 0)

        if not self.mask:
            x = self.net(x)
        else:
            x = self.net(x, mask_rend)

        x = self.fc(x)

        return x

    def forward(self, data):
        # print(data['transforms'].shape)
        # meta_data = {
        #     "camera_cx": data['camera_cx'][0],
        #     "camera_cy": data['camera_cy'][0],
        #     "camera_fx": data['camera_fx'][0],
        #     "camera_fy": data['camera_fy'][0],
        # }
        #
        # results = self.forward_(
        #     data['img'][0:1], data['depth'][0:1], data['transforms'], meta_data,
        #     data['model_points'][0], data['model_colors'][0], data['model_normals'][0]
        # )

        x = data['rend_x']
        mask = data['mask_x']
        # print(x.shape)

        if not self.mask:
            x = self.net(x)
        else:
            x = self.net(x, mask)

        results = self.fc(x)

        return results
