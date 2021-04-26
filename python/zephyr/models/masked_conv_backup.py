import os, time

import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader

from oriented_features.masked.masked_resnet import masked_resnet18
from oriented_features.masked.masked_comparison_resnet import masked_comparison_resnet
from oriented_features.image_utils.distance_transform import RadialDistanceTransform
from oriented_features.image_utils.edges import generate_distance_image
from oriented_features.projective_renderer import projective_render

from pytorch_lightning.core.lightning import LightningModule

from .base import BaseModel

class ConvolutionalPoseModel(BaseModel):

    def __init__(self, args, pretrained=True):
        super(ConvolutionalPoseModel, self).__init__(args)

        dist_max = args.dist_max
        self.dist_trans = RadialDistanceTransform(dist_max)

        self.img_model = masked_comparison_resnet(pretrained=True, num_classes=32, input_channels=[4,4])
        self.depth_model = masked_resnet18(pretrained=False, num_classes=32, input_channels=3)

        self.fc = torch.nn.Linear(64, 1)

    def forward_(self, img_obs, depth_obs, transforms, meta_data,
                model_points, model_colors, model_normals):

        # print('done')
        # time.sleep(10)
        # exit()
        with torch.no_grad():
            img_rend, depth_rend, mask_rend = projective_render(model_points, model_colors, model_normals,
                transforms, meta_data, gpu_id = img_obs.device, image_size = img_obs.shape[-2:], downsample=4)

        img_rend = img_rend.permute(0,3,1,2)
        depth_rend = depth_rend.unsqueeze(1)
        mask_rend = mask_rend.unsqueeze(1)

        edge_dist = generate_distance_image(depth_obs[0,0])

        img_obs = torch.cat([img_obs, edge_dist], dim=1)
        #depth_obs = torch.cat([depth_obs, edge_dist], dim=1)

        mask_dist, _ = self.dist_trans(mask_rend)
        img_rend = torch.cat([img_rend, mask_dist], dim=1)

        #depth_rend = torch.cat([depth_rend, mask_dist], dim=1)

        x_img = self.img_model(img_obs, img_rend, mask_rend)

        depth_obs = F.adaptive_avg_pool2d(depth_obs, depth_rend.shape[-2:])
        depth_dist = F.adaptive_avg_pool2d(mask_dist, depth_rend.shape[-2:])

        x_depth = torch.cat([depth_obs-depth_rend, depth_dist, mask_dist], dim=1)
        x_depth = self.depth_model(x_depth, mask_rend)

        return torch.relu(self.fc(torch.cat([x_img,x_depth], dim=-1)))

    def forward(self, data):
        # print(data['transforms'].shape)
        meta_data = {
            "camera_cx": data['camera_cx'][0],
            "camera_cy": data['camera_cy'][0],
            "camera_fx": data['camera_fx'][0],
            "camera_fy": data['camera_fy'][0],
        }

        results = self.forward_(
            data['img'][0:1], data['depth'][0:1], data['transforms'], meta_data,
            data['model_points'][0], data['model_colors'][0], data['model_normals'][0]
        )

        # print("results", results.shape)

        return results
        # print("forward")
        # for k, v in data.items():
        #     print(k)
        #     try:
        #         print(v.shape)
        #         print(v.dtype)
        #     except:
        #         print(len(v), v[0])
        # results = self.forward_(
        #     data['img'], data['depth'], data['']
        # )
