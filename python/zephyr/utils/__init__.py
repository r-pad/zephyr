import numpy as np
import torch

import torch


def to_np(x):
    if type(x) is np.ndarray:
        return x
        
    if torch.is_tensor(x):
        return x.detach().cpu().numpy()
    else:
        return x.detach().data.cpu().numpy()

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

def K2meta(cam_K):
    meta_data = {
        "camera_fx": cam_K[0,0],
        "camera_fy": cam_K[1,1],
        "camera_cx": cam_K[0,2],
        "camera_cy": cam_K[1,2],
        "camera_scale": 10000
    }
    return meta_data

def torch_norm_fast(tensor, axis):
    return torch.sqrt((tensor**2).sum(axis))

def depth2cloud(depth, mask, cam_K):
    h, w = depth.shape
    depth_mask_lin = mask.flatten()
    xmap, ymap = np.meshgrid(np.arange(w), np.arange(h))

    z = depth.flatten()[depth_mask_lin]
    x = xmap.flatten()[depth_mask_lin]
    y = ymap.flatten()[depth_mask_lin]

    z = z
    x = (x - cam_K[0,2]) * z / cam_K[0,0]
    y = (y - cam_K[1,2]) * z / cam_K[1,1]
    P_w = np.vstack((x, y, z)).T
    return P_w

def depth2xyz(depth, cam_K):
    h, w = depth.shape

    zmap = depth
    xmap, ymap = np.meshgrid(np.arange(w), np.arange(h))

    xmap = (xmap - cam_K[0,2]) * zmap / cam_K[0,0]
    ymap = (ymap - cam_K[1,2]) * zmap / cam_K[1,1]

    xyz = np.stack((xmap, ymap, zmap), axis=-1)

    return xyz

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


def getClosestTrans(transforms, mat_gt):
    quats = np.stack([quaternion_from_matrix(T) for T in transforms])
    trans = np.stack([T[:3,3] for T in transforms])
    gt_quats = quaternion_from_matrix(mat_gt)
    gt_trans = mat_gt[:3,3]

    q_diff = quatAngularDiffDot(quats, gt_quats)
    t_diff = np.linalg.norm(trans - gt_trans, axis=-1)

    t_dists = q_diff + t_diff
    min_idx = np.argmin(q_diff + t_diff)
    return min_idx

def dict_to(dictionary, device):
    for k,v in dictionary.items():
        if(type(v) is torch.Tensor):
            dictionary[k]=v.to(device)
