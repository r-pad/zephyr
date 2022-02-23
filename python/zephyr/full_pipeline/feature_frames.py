import numpy as np
import torch
from zephyr.normals import compute_normals
from zephyr.full_pipeline.keypoints import keypoints2Cloud

def featurize(img, depth, mask, meta_data, featurizer, return_normals = False):
    normals_down_sample = 1
    normals = compute_normals(depth.astype(np.double)[::normals_down_sample, ::normals_down_sample],
                              meta_data = meta_data,
                              smoothing_size=20)

    init_keypoints, init_features = featurizer(img, depth=depth,
        meta_data=meta_data, mask=mask)
    #init_features /= init_features.sum(axis=1).reshape(-1, 1)

    features = []
    keypoints = []
    keypoints_normals = []
    for kp, feat in zip(init_keypoints, init_features):
        norm = normals[int(kp.pt[1])//normals_down_sample, int(kp.pt[0])//normals_down_sample]
        if(~np.isnan(norm).any()):
            features.append(feat)
            keypoints.append(kp)
            keypoints_normals.append(norm)
    features = np.array(features)
    keypoints_normals = np.array(keypoints_normals)
    keypoints_ori = [kp.angle*np.pi/180. for kp in keypoints]
    cloud, _ = keypoints2Cloud(keypoints, depth, meta_data)
    frames = compute_frames(cloud,
                            keypoints_normals,
                            keypoints_ori)

    if(return_normals):
        return keypoints, features, cloud, frames, normals
    return keypoints, features, cloud, frames

def compute_basis(normal, orientation):
    u1 = normal / np.linalg.norm(normal)

    ez = np.array([0,0,1])
    v2 = np.array([np.cos(orientation), np.sin(orientation), 0])
    u2 = v2 - u1.dot(v2)/u1.dot(ez) * ez
    u2 /= np.linalg.norm(u2)
    u3 = np.cross(u1, u2)

    frame = np.stack([u1,u2,u3], axis=1)

    return frame

def compute_frames(points, normals, orientations):
    orientations = np.asarray(orientations)
    n = len(points)
    frames = np.zeros((n, 4, 4))

    frames[:, :3, 3] = points
    frames[:, 3, 3] = 1
    for i, (pt, n_vec, th) in enumerate(zip(points, normals, orientations)):
        # trans_mat = np.eye(4)
        # trans_mat[:3,:3] = compute_basis(n_vec, th)
        # trans_mat[:3,3] = pt
        # frames.append(trans_mat)
        frames[i, :3, :3] = compute_basis(n_vec, th)
    return frames

'''Vectorized version of the above 2 functions'''
def compute_frames(points, normals, orientations):
    orientations = np.asarray(orientations)
    n = len(points)
    frames = np.zeros((n, 4, 4))
    frames[:, :3, 3] = points
    frames[:, 3, 3] = 1

    u1 = normals / np.linalg.norm(normals, 2, axis=-1, keepdims=True)
    frames[:, :3, 0] = u1
    v2 = np.stack([np.cos(orientations), np.sin(orientations), np.zeros(n)], axis=1)
    ez = np.asarray([[0,0,1]])
    ez = np.tile(ez, (n, 1))
    u2 = v2 - (np.einsum('ik,ik->i', u1, v2) / u1[:, -1])[:, None] * ez
    u2 /= np.linalg.norm(u2, 2, axis=-1, keepdims=True)
    u3 = np.cross(u1, u2)

    frames[:, :3, 1] = u2
    frames[:, :3, 2] = u3

    return frames

def frame_transforms(frame_source, frame_target):
    R = frame_target[:3,:3].dot(frame_source[:3,:3].T)
    t = frame_target[:3,3] - R.dot(frame_source[:3,3])

    trans_mat = np.eye(4)
    trans_mat[:3,:3] = R
    trans_mat[:3,3] = t
    return trans_mat

def compute_transforms_iter(frames_source, frames_target):
    transforms = []
    for f_src, f_tgt in zip(frames_source, frames_target):
        transforms.append(frame_transforms(f_src, f_tgt))
    return transforms

def compute_transforms(frames_source, frames_target,
                             return_seperate = False):
    R_tgt = frames_target[:, :3, :3]
    R_src = frames_source[:, :3, :3]
    R = torch.einsum('ijk,imk->ijm', R_tgt, R_src)

    t_tgt = frames_target[:, :3, 3]
    t_src = frames_source[:, :3, 3]
    t = t_tgt - torch.einsum('ijk,ik->ij', R, t_src)

    if(return_seperate):
        return R, t

    trans_mats = torch.zeros_like(frames_source)
    trans_mats[:, :3, :3] = R
    trans_mats[:, :3, 3] = t
    trans_mats[:,3,3] = 1.0
    return trans_mats
