import open3d as o3d

import time
import numpy as np
import cv2
from scipy import ndimage

from . import depth2cloud

def point2pointICP(src_pc, tgt_pc, mat_max, icp_max_dist):
    source = o3d.geometry.PointCloud()
    target = o3d.geometry.PointCloud()
    source.points = o3d.utility.Vector3dVector(src_pc)
    target.points = o3d.utility.Vector3dVector(tgt_pc)
    reg_p2p = o3d.registration.registration_icp(
        source, target, icp_max_dist, mat_max,
        o3d.registration.TransformationEstimationPointToPoint(),
    )
    mat_refine = reg_p2p.transformation
    return mat_refine

def icpRefinement(depth, uv_this, mat, cam_K, model_points, inpaint_depth=False, icp_max_dist=0.01):
    l, u = uv_this.min(0)
    r, d = uv_this.max(0)

    d = min(d + int((d-u) / 2), depth.shape[0])
    r = min(r + int((r-l) / 2), depth.shape[1])
    l = max(l - int((r-l) / 2), 0)
    u = max(u - int((d-u) / 2), 0)
    bbox_max = np.zeros_like(depth)
    bbox_max[u:d, l:r] = 1
    if inpaint_depth:
        depth = cv2.inpaint(depth, (depth == 0).astype(np.uint8), 2, cv2.INPAINT_NS)
        depth = ndimage.gaussian_filter(depth,2)
    else:
        bbox_max[depth==0] = 0
    bbox_max = bbox_max.astype(bool)
    crop_scene_pc = depth2cloud(depth, bbox_max, cam_K)

    t1 = time.time()
    mat_icp = point2pointICP(model_points, crop_scene_pc, mat, icp_max_dist = icp_max_dist)
    t2 = time.time()

    time_icp = t2 - t1

    return mat_icp, time_icp
