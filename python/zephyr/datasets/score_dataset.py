import os, copy
import cv2
from functools import partial

import numpy as np
import torch
import torchvision
from torch.utils.data import Dataset

from zephyr.data_util import to_np, vectorize, img2uint8
from zephyr.utils import torch_norm_fast
from zephyr.utils.mask_edge import getRendEdgeScore
from zephyr.utils.edges import generate_distance_image
from zephyr.normals import compute_normals
from zephyr.utils.timer import TorchTimer

try:
    from zephyr.datasets.bop_raw_dataset import BopRawDataset
except ImportError:
    pass
from zephyr.datasets.prep_dataset import PrepDataset

IMPORTANCE_ORDER = [
    28, 27, 32, 33, 36, 35, 29, 16, 26, 22, 13, 4, 26, 21, 22
]

class ScoreDataset(Dataset):
    def __init__(self, datapoints, dataset_root, dataset_name, args, mode='train', timing = False):
        self.args = args
        self.datapoints = datapoints
        self.dataset_root = dataset_root
        self.dataset_name = dataset_name
        self.mode = mode
        self.return_full_data = False

        self.feature_size = args.feature_size
        self.norm_cos_weight = args.norm_cos_weight
        self.top_n_feat = args.top_n_feat
        self.max_hypos = args.max_hypos
        self.ppf_only = args.ppf_only
        self.n_ppf_hypos = args.n_ppf_hypos
        self.n_sift_hypos = args.n_sift_hypos
        self.use_mask_test = args.use_mask_test

        if args.raw_bop_dataset:
            self.loader = BopRawDataset(
                args.bop_root, self.dataset_name, args.split, args.split_name, args.ppf_result_file, no_sift=args.ppf_only, no_ppf=args.sift_only
                )
        else:
            self.loader = PrepDataset(
                self.dataset_root, self.feature_size
            )

        self.dim_point = 0
        self.dim_render = 0
        self.dim_agg = 0

        # About timing
        self.timing = timing
        self.timing_list = []

        if args.model_name == "maskconv":
            print("Using Point Render dataset")
            self.return_rend, self.return_points, self.return_agg = True, True, False
        else:
            self.return_rend = False
            if args.dataset == "feat":
                print("Using Agg Dataset")
                self.return_points, self.return_agg = False, True
            else: # Use PointNet dataset
                if "mix" in args.dataset:
                    print("Using Mix Dataset")
                    self.return_points, self.return_agg = True, True
                else:
                    print("Using PointNet Dataset")
                    self.return_points, self.return_agg = True, False

        '''For aggregated features Data'''
        if self.return_agg:
            self.std = None
            self.mean = None

            self.feature_inliers = True
            self.use_hsv = True
            self.normalize = True
            self.fs_thresh = 0.02

            if args.selected_features is not None:
                self.selected_features = args.selected_features
                print("Using feature indices:", self.selected_features)
            elif self.top_n_feat is not None:
                self.selected_features = IMPORTANCE_ORDER[:self.top_n_feat]
                print("ScoreDataset: Using top features N =", self.top_n_feat)
                print("Using feature indices:", self.selected_features)
                args.selected_features = self.selected_features
            else:
                self.selected_features = list(range(39))
                print("Using all aggregated features")
                args.selected_features = self.selected_features

            self.dim_agg = len(self.selected_features)

            self.vectorize = partial(vectorize,
                use_hsv=self.use_hsv,
                feature_inliers=self.feature_inliers,
                norm_cos_weight=self.norm_cos_weight,
                fs_thresh=self.fs_thresh
                )

            self.agg_cache = [None for _ in range(len(self.datapoints))]

        '''For PointNet Data'''
        self.point_x_labels = []
        if self.return_points:
            self.max_points = args.max_points
            args.xyz_channel = [] # indices of point_x channels that define coordinates
            args.model_channel = [] # indices of point_x channels that are specific to the object model

            '''Mask channel'''
            num_features = 0
            # valid_proj.unsqueeze(-1).float(),
            # valid_depth.unsqueeze(-1).float(),

            if not self.args.no_valid_proj:
                self.point_x_labels += ['valid_proj']
                num_features += 1
            if not self.args.no_valid_depth:
                self.point_x_labels += ["valid_depth"]
                num_features += 1

            '''XYZ channel'''
            self.uvd, self.uv = False, False
            if "uvd" in args.dataset:
                self.uvd = True
                args.xyz_channel = list(range(num_features, num_features + 3))
                num_features +=3
                self.point_x_labels += ['u', 'v', 'd']
            elif "uv" in args.dataset:
                self.uv = True
                args.xyz_channel = list(range(num_features, num_features + 2))
                num_features += 2
                self.point_x_labels += ['u', 'v']
            else:
                num_features += 0
            args.model_channel += args.xyz_channel
            num_non_data = num_features

            '''Data channel'''
            if "cos" in args.dataset:
                self.point_x_labels += ['cam_norm_cos']

            self.RGB, self.HSV, self.D, self.diff, self.cos, self.edge, self.ppfscore, self.norm_cos = \
                False, False, False, False, False, False, False, False
            if "RGB" in args.dataset:
                self.RGB, self.HSV = True, False
                args.model_channel += list(range(num_features, num_features + 3))
                num_features += 6
                self.point_x_labels += ['R_diff', 'G_diff', 'B_diff'] if "diff" in args.dataset else ["R1", "G1", "B1", "R2", "G2", "B2"]
            elif "HSV" in args.dataset:
                self.RGB, self.HSV = True, True
                args.model_channel += list(range(num_features, num_features + 3))
                num_features += 6
                self.point_x_labels += ['H_diff', 'S_diff', 'V_diff'] if "diff" in args.dataset else ["H1", "S1", "V1", "H2", "S2", "V2"]
            if "D" in args.dataset:
                self.D = True
                args.model_channel += list(range(num_features, num_features + 1))
                num_features += 2
                self.point_x_labels += ["D_diff"] if "diff" in args.dataset else ["D1", "D2"]

            if "diff" in args.dataset:
                self.diff = True
                num_features = num_non_data + (num_features-num_non_data) // 2

            if "cos" in args.dataset:
                self.cos = True
                num_features += 1

            if "edge" in args.dataset:
                self.edge = True
                self.edgecos = "edgecos" in args.dataset
                self.edgexnor = "edgexnor" in args.dataset
                num_features += 1 if (self.edgecos or self.edgexnor) else 2
                if self.edgecos:
                    self.point_x_labels += ['obs_edge_score']
                elif self.edgexnor:
                    self.point_x_labels += ['edge_xnor']
                else:
                    self.point_x_labels += ['obs_edge_score', "rend_edge_score"]
            if "ppfscore" in args.dataset:
                self.ppfscore = True
                num_features += 1
                self.point_x_labels += ['ppf_score']

            if "norm" in args.dataset:
                self.norm_cos = True
                num_features += 1
                self.point_x_labels += ['norm_cos']

            self.seg_mask = False
            if "seg" in args.dataset:
                self.seg_mask = True
                num_features += 1
                self.point_x_labels += ['mask', "mask_edge"]

            self.dim_point = num_features

            '''Train/Test specific config'''
            if self.mode == 'train':
                print("Initializating training dataset", self.point_x_labels)
                self.cojitter = args.cojitter
                self.drop_ratio = args.drop_ratio
                self.uv_rot = args.uv_rot
            else:
                print("Initializating %s dataset" % mode, self.point_x_labels)
                self.cojitter = False
                self.drop_ratio = 0
                self.uv_rot = False

            self.transform = torchvision.transforms.Compose([
                torchvision.transforms.ToPILImage(),
                torchvision.transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.05),
                torchvision.transforms.ToTensor(),
            ])
            if self.cojitter:
                self.transform_cojitter = torchvision.transforms.Compose([
                    torchvision.transforms.ToPILImage(),
                    torchvision.transforms.ColorJitter(brightness=0.5, contrast=0.5, saturation=0.5, hue=0.5),
                    torchvision.transforms.ToTensor(),
                ])
                print("ScorePointnetDataset: Using cojitter")

            if self.return_rend:
                self.dim_render = self.dim_point - 1

    def __len__(self):
        return len(self.datapoints)

    def setNormalization(self, var, mean):
        var = torch.from_numpy(np.asarray(var))
        mean = torch.from_numpy(np.asarray(mean))
        self.std = torch.sqrt(var[self.selected_features]).float()
        self.mean = mean[self.selected_features].float()

    '''Return [n_hypo, n_features]'''
    def getAggData(self, data):
        x = self.vectorize(data)

        x = x[:, self.selected_features]

        if self.normalize:
            x = (x-self.mean)/self.std
        return x

    '''Return [n_hypo, n_points, n_features]'''
    def getPointNetData(self, data, return_uv_original=False):
        with TorchTimer("Data convert 1", agg_list=self.timing_list, timing = self.timing, verbose=False):
            img = data['img'].float() # float [0, 1]
            depth = data['depth'].float()

            if "pbr" in self.dataset_root and self.mode == "train":
                # print("blur depth image")
                depth = depth * (torch.ones_like(depth) + 0.003 * torch.randn_like(depth))

            transforms = data['transforms'].float()
            model_points = data['model_points'].float()
            model_colors = data['model_colors'].float() # float [0, 1]
            model_normals = data['model_normals'].float()
            meta_data = data['meta_data']

        with TorchTimer("Transform and project", agg_list=self.timing_list, timing = self.timing, verbose=False):
            # Transform and project point cloud
            trans_pts = torch.einsum('ijk,mk->imj', transforms[:,:3,:3], model_points) + transforms[:,:3,3].unsqueeze(1)
            f_cam = torch.tensor([meta_data['camera_fx'], meta_data['camera_fy']])
            c_cam = torch.tensor([meta_data['camera_cx'], meta_data['camera_cy']])
            proj_pts = trans_pts[:,:,:2]/trans_pts[:,:,2:]*f_cam + c_cam
            uv = proj_pts.long()
            invalid_proj = (uv[:,:,1]>=img.shape[0]) + (uv[:,:,1]<0) \
                + (uv[:,:,0]>=img.shape[1]) + (uv[:,:,0]< 0)
            uv[invalid_proj] = 0

            # Projected depth
            proj_depth = trans_pts[:,:,-1]

        '''Jitter the color as data augmentation'''
        if self.mode == "train":
            img = img.permute(2, 0, 1) # (H, W, C) to (C, H, W)
            img = self.transform(img)
            img = img.permute(1, 2, 0) # (C, H, W) to (H, W, C)

            if self.cojitter:
                H, W, C = img.shape # (H, W, C)
                N, _ = model_colors.shape
                data_cojitter = torch.cat([
                    img.reshape((1, -1, 3)),
                    model_colors.reshape((1, -1, 3))
                ], dim=1)

                data_cojitter = data_cojitter.permute(2, 0, 1)
                cojittered = self.transform_cojitter(data_cojitter)
                cojittered = cojittered.permute(1, 2, 0)

                img = cojittered[0, :H*W, :].reshape((H, W, C))
                model_colors = cojittered[0, H*W:, :].reshape((N, C))

        # RGb to HSV
        with TorchTimer("RGB to HSV", agg_list=self.timing_list, timing = self.timing, verbose=False):
            if self.HSV:
                with np.errstate(divide='ignore'):
                    img_rgb = img2uint8(to_np(img))
                    # img_hsv = rgb2hsv(img_rgb) # this will convert it to range [0, 1]
                    img_hsv = cv2.cvtColor(img_rgb,cv2.COLOR_RGB2HSV)
                    img_hsv = img_hsv.astype(float) / 255.0
                    img = torch.from_numpy(img_hsv).to(img.device).float()
                    model_colors_rgb = img2uint8(np.expand_dims(to_np(model_colors), 0))
                    # model_colors_hsv = rgb2hsv(model_colors_rgb)[0]
                    model_colors_hsv = cv2.cvtColor(model_colors_rgb,cv2.COLOR_RGB2HSV)[0]
                    model_colors_hsv = model_colors_hsv.astype(float) / 255.0
                    model_colors = torch.from_numpy(model_colors_hsv).to(model_colors.device).float()

        # Sample the observed HSVD
        with TorchTimer("Sample obvervation", agg_list=self.timing_list, timing = self.timing, verbose=False):
            obs_color = img[uv[:,:,1], uv[:,:,0], :]
            obs_depth = depth[uv[:,:,1], uv[:,:,0]]

        with TorchTimer("Hypo Pruning", agg_list=self.timing_list, timing = self.timing, verbose=False):
            if self.args.inconst_ratio_th is not None and self.mode == "test":
                d_diff = proj_depth - obs_depth
                n_points = model_points.shape[0]
                invalid_count = (d_diff < -0.02).sum(-1).float()
                invalid_ratio = invalid_count / n_points
                th = self.args.inconst_ratio_th
                idx = invalid_ratio < (th/100.0)
                idx[-1] = True

                # At least preserve some non-oracle hypos
                if idx.sum() == 1:
                    idx[0] = True

                pruning_mask = idx

                transforms = transforms[idx]
                trans_pts = trans_pts[idx]
                obs_color = obs_color[idx]
                obs_depth = obs_depth[idx]
                uv = uv[idx]
                invalid_proj = invalid_proj[idx]
                proj_depth = proj_depth[idx]

                self.SelectDataByIdx(data, idx)

            uv_original = copy.deepcopy(uv)
            data['uv_original'] = uv_original

        # Transform normals
        with TorchTimer("Transform and project 2", agg_list=self.timing_list, timing = self.timing, verbose=False):
            trans_norms = torch.einsum('ijk,mk->imj', transforms[:,:3,:3], model_normals)
            cam_norm_cos = (- trans_pts * trans_norms).sum(-1) / (torch_norm_fast(trans_pts, -1) * torch_norm_fast(trans_norms, -1))
            valid_norm = cam_norm_cos > 0
            valid_proj = valid_norm * torch.bitwise_not(invalid_proj)
            
        data['valid_proj'] = valid_proj

        # x = []
        x = model_points.new_empty((len(transforms), len(model_points), self.dim_point))
        idx_feat = 0

        with TorchTimer("Valid proj/depth", agg_list=self.timing_list, timing = self.timing, verbose=False):
            valid_depth = obs_depth > 0
            '''Mask channel'''
            if not self.args.no_valid_proj:
                # x += [valid_proj.unsqueeze(-1).float()]
                x[:, :, idx_feat] = valid_proj.float()
                idx_feat += 1
            if not self.args.no_valid_depth:
                # x += [valid_depth.unsqueeze(-1).float()]
                x[:, :, idx_feat] = valid_depth.float()
                idx_feat += 1

        '''XYZ channel'''
        with TorchTimer("Normalize uv", agg_list=self.timing_list, timing = self.timing, verbose=False):
            if self.uv or self.uvd:
                uv = uv.float()
                uv_mean = uv.mean(dim=1, keepdim=True)
                uv_std = uv.std(dim=1, keepdim=True)
                uv = (uv - uv_mean) / uv_std
                if self.uv_rot:
                    n_hypo, n_point, n_coord = uv.shape

                    '''random flip'''
                    flip_mat = torch.rand((n_hypo, 1, n_coord)) > 0.5
                    flip_mat = (flip_mat.type(uv.dtype) - 0.5) * 2
                    uv = uv * flip_mat

                    '''random rotation'''
                    rot_mat = torch.rand((n_hypo, 1, 1)) * 2 * np.pi
                    rot_mat = torch.cat([
                        torch.cos(rot_mat), -torch.sin(rot_mat),
                        torch.sin(rot_mat), torch.cos(rot_mat)
                    ], 2).reshape((-1, 1, 2, 2))
                    uv = uv.unsqueeze(-1)
                    uv = torch.matmul(rot_mat, uv)
                    uv = uv.squeeze()

                # x += [uv]
                x[:, :, idx_feat:idx_feat+2] = uv
                idx_feat += 2
                if self.uvd:
                    d_diff = proj_depth.unsqueeze(-1) - obs_depth.unsqueeze(-1)
                    d_diff = (d_diff - d_diff.mean(dim=1, keepdim=True)) / d_diff.std(dim=1, keepdim=True)
                    # x += [d_diff]
                    x[:, :, idx_feat:idx_feat+1] = d_diff
                    idx_feat += 1

        '''Point data channel'''
        if self.cos:
            # x += [cam_norm_cos.unsqueeze(-1).float()]
            x[:, :, idx_feat] = cam_norm_cos.float()
            idx_feat += 1

        with TorchTimer("Compute RGBD/HSVD diff", agg_list=self.timing_list, timing = self.timing, verbose=False):
            if self.RGB or self.HSV:
                if self.diff:
                    color_diff = model_colors.unsqueeze(0).expand(obs_color.shape) - obs_color
                    if self.HSV:
                        color_diff[:,:,0] = color_diff[:,:,0].abs()
                        color_diff[:,:,0] = np.minimum(color_diff[:,:,0], 1-color_diff[:,:,0])
                    # x += [color_diff]
                    x[:, :, idx_feat:idx_feat+3] = color_diff
                    idx_feat += 3
                else:
                    # x += [model_colors.unsqueeze(0).expand(obs_color.shape), obs_color]
                    x[:, :, idx_feat:idx_feat+3] = model_colors.unsqueeze(0).expand(obs_color.shape)
                    idx_feat += 3
                    x[:, :, idx_feat:idx_feat+3] = obs_color
                    idx_feat += 3

            if self.D:
                if self.diff:
                    # x += [proj_depth.unsqueeze(-1) - obs_depth.unsqueeze(-1)]
                    x[:, :, idx_feat] = proj_depth - obs_depth
                    idx_feat += 1
                else:
                    # x += [proj_depth.unsqueeze(-1), obs_depth.unsqueeze(-1)]
                    x[:, :, idx_feat] = proj_depth
                    idx_feat += 1
                    x[:, :, idx_feat] = obs_depth
                    idx_feat += 1

        '''Edge channel'''
        with TorchTimer("Edge", agg_list=self.timing_list, timing = self.timing, verbose=False):
            if self.edge:
                '''Observed edges'''
                if "depth_for_edge" in data:
                    depth_for_edge = data['depth_for_edge']
                    # print("Using depth_for_edge", depth_for_edge.min(), depth_for_edge.max())
                else:
                    depth_for_edge = depth

                with TorchTimer("generate_distance_image", agg_list=self.timing_list, timing = self.timing, verbose=False):
                    edge_obs = generate_distance_image(depth_for_edge, canny_l=20, canny_h=50)[0,0]

                with TorchTimer("Edge sampling", agg_list=self.timing_list, timing = self.timing, verbose=False):
                    uv = copy.deepcopy(uv_original) # Re-fetch the uv as it is changed before
                    edge_score_obs = edge_obs[uv[:,:,1], uv[:,:,0]]
                    edge_score_obs = torch.exp(-edge_score_obs / 24)

                '''Projected edges'''

                with TorchTimer("getRendEdgeScore", agg_list=self.timing_list, timing = self.timing, verbose=False):
                    if "edge_score_rend" in data:
                        edge_score_rend = data['edge_score_rend']
                    else:
                        with torch.no_grad():
                            edge_score_rend = getRendEdgeScore(img.to(self.args.edge_gpu), uv_original.to(self.args.edge_gpu)).to(uv_original.device)

                '''Normalized edge scores'''
                edge_score_rend = edge_score_rend / edge_score_rend.max(1, keepdim=True)[0]
                # edge_score_obs = torch.exp(-edge_score_obs / )

                if self.edgexnor:
                    edge_score = edge_score_rend * edge_score_obs + (1 - edge_score_rend) * (1 - edge_score_obs)
                    # x += [edge_score.unsqueeze(-1)]
                    x[:, :, idx_feat] = edge_score
                    idx_feat += 1
                elif self.edgecos:
                    # x += [edge_score_obs.unsqueeze(-1)]
                    x[:, :, idx_feat] = edge_score_obs
                    idx_feat += 1
                else:
                    # x += [edge_score_obs.unsqueeze(-1)]
                    # x += [edge_score_rend.unsqueeze(-1)]
                    x[:, :, idx_feat] = edge_score_obs
                    idx_feat += 1
                    x[:, :, idx_feat] = edge_score_rend
                    idx_feat += 1

        if self.args.camera_scale is not None:
            meta_data['camera_scale'] = self.args.camera_scale
            
        '''Use the cos of the angle between observed and rendered normal vectors'''
        with TorchTimer("Normal vector", agg_list=self.timing_list, timing = self.timing, verbose=False):
            if self.norm_cos:
                norm_downsample = self.args.norm_downsample
                uv = uv_original # Re-fetch the uv as it is changed before
                normals = compute_normals(to_np(depth)[::norm_downsample, ::norm_downsample].astype(np.double), meta_data = meta_data)
                normals = torch.from_numpy(normals).float()
                scene_normals_proj = normals[uv[:,:,1]//norm_downsample, uv[:,:,0]//norm_downsample]
                model_normals_proj = trans_norms
                norm_cos = (scene_normals_proj * model_normals_proj).sum(dim=-1) / (torch_norm_fast(scene_normals_proj, -1) * torch_norm_fast(model_normals_proj, -1))
                norm_cos[norm_cos != norm_cos] = 0
                # x += [norm_cos.unsqueeze(-1).float()]
                x[:, :, idx_feat] = norm_cos.float()
                idx_feat += 1

        # with TorchTimer("torch.cat()", agg_list=self.timing_list, timing = self.timing, verbose=False):
        #     x = torch.cat(x, dim=-1)
        # print(x.shape)

        if self.args.hard_mask:
            x[~valid_proj.bool()]=0

        '''Sample the points'''
        if self.drop_ratio >= 0 and self.mode == 'train':
            n_hypo = x.shape[0]
            n_point = x.shape[1]
            n_point_kept = int((1.0-self.drop_ratio) * n_point)

            if self.max_points is not None and n_point_kept > self.max_points:
                n_point_kept = self.max_points

            idx = []
            for i in range(n_hypo):
                idx.append(torch.randperm(n_point)[:n_point_kept].unsqueeze(0))
            idx = torch.cat(idx, dim=0)
            x = x[torch.arange(n_hypo).unsqueeze(1).expand(n_hypo, n_point_kept), idx]
            uv_sampled = uv_original[torch.arange(n_hypo).unsqueeze(1).expand(n_hypo, n_point_kept), idx]
        else:
            uv_sampled = uv_original

        if return_uv_original:
            return x, uv_sampled
        else:
            return x

    def getPointRenderData(self, data):
        point_x, uv = self.getPointNetData(data, True)

        crop_size = 96
        pad_size = 2
        n_hypo = uv.shape[0]
        n_point = uv.shape[1]

        span_min = pad_size
        span_max = crop_size - pad_size

        mask_index = [0]
        # data_index = [0, 1] + list(range(4, point_x.shape[2]))
        data_index = list(range(point_x.shape[2]))

        n_feat = len(data_index)

        point_mask = point_x[:, :, mask_index].bool()
        point_data = point_x[:, :, data_index]

        uv = uv.float()
        uv_max = uv.max(dim=1, keepdim=True)[0]
        uv_min = uv.min(dim=1, keepdim=True)[0]

        uv_center = (uv_max + uv_min) / 2.0
        uv_radius = (uv_max - uv_min).max(-1, True)[0] / 2.0

        uv_norm = (uv - uv_center) / uv_radius # range in [-1, 1]
        uv_resize = (uv_norm + 1) / 2 * (span_max - span_min) + span_min
        uv_resize = uv_resize.long()
        u = uv_resize[:, :, 0]
        v = uv_resize[:, :, 1]

        feature_map = torch.zeros(n_hypo, n_feat, crop_size, crop_size)
        t = torch.arange(n_hypo).view(-1,1).repeat(1, n_point)
        u = u.reshape(-1)[point_mask.view(-1)]
        v = v.reshape(-1)[point_mask.view(-1)]
        t = t.view(-1)[point_mask.view(-1)]

        feature_map[t.view(-1), :, v.view(-1), u.view(-1)] = point_data.view(-1, n_feat)[point_mask.view(-1)]
        mask_map = feature_map[:, 0:1, :, :]
        data_map = feature_map[:, 1:, :, :]

        return mask_map, data_map

    def SelectDataByIdx(self, data, idx):
        data['transforms'] = data['transforms'][idx]
        data['pp_err'] = data['pp_err'][idx]
        if "edge_score_rend" in data:
            data['edge_score_rend'] = data['edge_score_rend'][idx]
        return data

    def __getitem__(self, idx):
        dp = self.datapoints[idx]
        to_return = {"object_id": dp[0], "scene_id": dp[1], "im_id": dp[2]}
        obj_id = dp[0]
        scene_id = dp[1]
        im_id = dp[2]

        '''If only used aggregated features, return the cached one'''
        if self.return_agg and not self.return_points and self.agg_cache[idx] is not None:
            to_return['agg_x'], to_return['pp_err'], to_return['transforms'] = self.agg_cache[idx]
            return to_return

        data = self.loader.loadData(*dp)

        assert len(data['pp_err']) == 101 or len(data['pp_err']) == 1101 or len(data['pp_err']) == 301

        assert not (self.args.ppf_only and self.args.sift_only)
        if self.args.ppf_only:
            assert len(data['pp_err']) >= self.args.n_ppf_hypos + 1
            idx = list(np.arange(self.args.n_ppf_hypos)) + [-1]
            self.SelectDataByIdx(data, idx)

        if self.args.sift_only:
            assert len(data['pp_err']) >= self.args.n_ppf_hypos + self.args.n_sift_hypos + 1
            idx = list(range(self.n_ppf_hypos, self.n_ppf_hypos+self.n_sift_hypos)) + [-1]
            data = self.SelectDataByIdx(data, idx)

        '''Sample the hypotheses'''
        point_x = self.getPointNetData(data)

        n_hypo = len(point_x)
        to_return['object_id'] = to_return['object_id'].repeat(n_hypo)
        to_return['scene_id'] = to_return['scene_id'].repeat(n_hypo)
        to_return['im_id'] = to_return['im_id'].repeat(n_hypo)
        to_return['pp_err'] = data['pp_err'].reshape(-1)
        to_return['transforms'] = data['transforms']

        if self.return_agg:
            to_return['agg_x'] = self.getAggData(data)
            self.agg_cache[idx] = (to_return['agg_x'], to_return['pp_err'], to_return['transforms'])
        if self.return_points:
            if self.return_rend:
                to_return['rend_mask'], to_return['x_rend'] = self.getPointRenderData(data)
                to_return['mask_x'] = to_return['rend_mask']
                to_return['rend_x'] = to_return['x_rend']
            else:
                to_return['point_x'] = point_x

        # print("to_return['pp_err']", to_return['pp_err'])
        # print("to_return['pp_err']", to_return['pp_err'].shape)
        # print("to_return['transforms']", to_return['transforms'].shape)
        # print("to_return['point_x']", to_return['point_x'].shape)

        to_return['dataset_i'] = 0

        # For ICP post-processing
        to_return['depth'] = data['depth']
        to_return['meta_data'] = data['meta_data']
        to_return['uv_original'] = data['uv_original']
        to_return['model_points'] = data['model_points']

        return to_return
