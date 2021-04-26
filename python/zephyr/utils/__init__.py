import numpy as np
import torch

import torch

def meta2K(meta_data):
    if type(meta_data['camera_fx']) is torch.Tensor:
        cam_K = np.asarray([
            [meta_data['camera_fx'].item(), 0, meta_data['camera_cx'].item()],
            [0, meta_data['camera_fy'].item(), meta_data['camera_cy'].item()],
            [0, 0, 1]
        ])
    else:
        cam_K = np.asarray([
            [meta_data['camera_fx'], 0, meta_data['camera_cx']],
            [0, meta_data['camera_fy'], meta_data['camera_cy']],
            [0, 0, 1]
        ])

    return cam_K

def torch_norm_fast(tensor, axis):
    return torch.sqrt((tensor**2).sum(axis))

def depth2cloud(depth, mask, cam_K):
    h, w = depth.shape
    depth_mask_lin = mask.flatten()
    ymap, xmap = np.meshgrid(np.arange(w), np.arange(h))

    z = depth.flatten()[depth_mask_lin]
    x = ymap.flatten()[depth_mask_lin]
    y = xmap.flatten()[depth_mask_lin]

    z = z
    x = (x - cam_K[0,2]) * z / cam_K[0,0]
    y = (y - cam_K[1,2]) * z / cam_K[1,1]
    P_w = np.vstack((x, y, z)).T
    return P_w

def projectPointsUv(transforms, model_points, meta_data):
    import torch
    if type(transforms) is torch.Tensor:
        transforms = transforms.float()
        model_points = model_points.float()
        trans_pts = torch.einsum('ijk,mk->imj', transforms[:,:3,:3], model_points) + transforms[:,:3,3].unsqueeze(1)
        f_cam = torch.tensor([meta_data['camera_fx'], meta_data['camera_fy']])
        c_cam = torch.tensor([meta_data['camera_cx'], meta_data['camera_cy']])
        proj_pts = trans_pts[:,:,:2]/trans_pts[:,:,2:]*f_cam + c_cam
        uv = torch.round(proj_pts).long()
    else:
        trans_pts = np.einsum('ijk,mk->imj', transforms[:,:3,:3], model_points) + np.expand_dims(transforms[:,:3,3], 1)
        f_cam = np.asarray([meta_data['camera_fx'], meta_data['camera_fy']])
        c_cam = np.asarray([meta_data['camera_cx'], meta_data['camera_cy']])
        proj_pts = trans_pts[:,:,:2]/trans_pts[:,:,2:]*f_cam + c_cam
        uv = proj_pts.round().astype(int)
    return uv
