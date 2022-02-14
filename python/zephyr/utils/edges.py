import numpy as np
import cv2
import torch
from scipy import ndimage

from zephyr.utils import to_np

def generate_distance_image(depth, depth_clip_up=None, canny_l=100, canny_h=200):
    if(type(depth) is torch.Tensor):
        device = depth.device
        depth = to_np(depth)

    '''clip the depth image so that it can have finer resolution in the close region'''
    if depth_clip_up is None:
        img_d = depth / depth.max()
    else:
        img_d = depth.clip(0, depth_clip_up)
        img_d = img_d / img_d.max()

    edge = cv2.Canny(np.round(img_d * 255).astype(np.uint8), canny_l, canny_h)
    dist = ndimage.distance_transform_edt(~edge)
    dist = torch.from_numpy(dist).view(1,1,*dist.shape).float().to(device)
    return dist
