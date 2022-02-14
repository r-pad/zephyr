import numpy as np
import cv2

def mask2Keypoints(mask, step_size = 1, feature_size = None):
    if(feature_size is None):
        feature_size = step_size

    keypoints = []
    for y in range(0, mask.shape[0], step_size):
        for x in range(0, mask.shape[1], step_size):
            if(mask[y,x]):
                keypoints.append(cv2.KeyPoint(x=x, y=y, _size=feature_size))
   
    return keypoints

def keypoints2Cloud(keypoints, depth, meta_data):
    y_size, x_size = depth.shape[:2]
    ymap, xmap = np.meshgrid(np.arange(x_size), np.arange(y_size))

    choose = np.ravel_multi_index(np.array([(pt.pt[1],pt.pt[0]) for pt in keypoints]).astype(int).T,
                                  (y_size, x_size))

    depth_choose = depth.flatten()[choose][:, np.newaxis].astype(np.float32)
    xmap_choose = xmap.flatten()[choose][:, np.newaxis].astype(np.float32)
    ymap_choose = ymap.flatten()[choose][:, np.newaxis].astype(np.float32)
    z = depth_choose / meta_data['camera_scale']
    x = (ymap_choose - meta_data['camera_cx']) * z / meta_data['camera_fx']
    y = (xmap_choose - meta_data['camera_cy']) * z / meta_data['camera_fy']
    cloud = np.concatenate((x, y, z), axis=1)
    return cloud, choose

def keypoints2Indices(keypoints, size = None, return_map = False):
    #if(len(keypoints) > 0 and type(keypoints[0]) == cv2.KeyPoint):
    #    keypoints = cv2.KeyPoint_convert(keypoints)
    if(size is None):
        size = np.max(keypoints, axis=0)
    indices = keypoints[:,1].round().astype(int)*size[0] + keypoints[:,0].round().astype(int)
    if(return_map):
        return {v:k for k,v in enumerate(indices)}
    return indices

def coord2Cloud(coords, depth, meta_data):
    y_size, x_size = depth.shape[:2]
    ymap, xmap = np.meshgrid(np.arange(x_size), np.arange(y_size))

    choose = np.ravel_multi_index(np.flip(coords, axis=1).T, (y_size, x_size))

    depth_choose = depth.flatten()[choose][:, np.newaxis].astype(np.float32)
    xmap_choose = xmap.flatten()[choose][:, np.newaxis].astype(np.float32)
    ymap_choose = ymap.flatten()[choose][:, np.newaxis].astype(np.float32)
    z = depth_choose / meta_data['camera_scale']
    x = (ymap_choose - meta_data['camera_cx']) * z / meta_data['camera_fx']
    y = (xmap_choose - meta_data['camera_cy']) * z / meta_data['camera_fy']
    cloud = np.concatenate((x, y, z), axis=1)
    return cloud, choose
