import numpy as np
import torch
import torch.nn as nn
import cv2
from scipy import ndimage
import math

def getRendEdgeScore(img, uv, downsample = 4):
    img_size = img.shape
    n_hypo = uv.shape[0]
    n_point = uv.shape[1]
    # Get rendered mask
    uv = uv // downsample
    edge_rend = torch.zeros(n_hypo, img_size[0] // downsample, img_size[1] // downsample).float().to(uv.device)
    first_idx = torch.arange(n_hypo).repeat_interleave(n_point)
    edge_rend[first_idx, uv[:, :, 1].flatten(), uv[:, :, 0].flatten()] = 1
    # mask to edge
    edge_rend = mask2edge(edge_rend.unsqueeze(1)).float()
    # BLur the edge image
    edge_blur = get_gaussian_kernel(kernel_size=11, sigma=6, channels=1).to(uv.device)
    edge_rend = edge_blur(edge_rend)
    first_idx = torch.arange(n_hypo).reshape((-1, 1)).expand(-1, n_point)
    edge_score_rend = edge_rend[first_idx, 0, uv[:,:,1], uv[:, :, 0]]

    return edge_score_rend

def get_gaussian_kernel(kernel_size=3, sigma=2, channels=3):
    # Create a x, y coordinate grid of shape (kernel_size, kernel_size, 2)
    x_coord = torch.arange(kernel_size)
    x_grid = x_coord.repeat(kernel_size).view(kernel_size, kernel_size)
    y_grid = x_grid.t()
    xy_grid = torch.stack([x_grid, y_grid], dim=-1).float()

    mean = (kernel_size - 1)/2.
    variance = sigma**2.

    # Calculate the 2-dimensional gaussian kernel which is
    # the product of two gaussian distributions for two different
    # variables (in this case called x and y)
    gaussian_kernel = (1./(2.*math.pi*variance)) *\
                      torch.exp(
                          -torch.sum((xy_grid - mean)**2., dim=-1) /\
                          (2*variance)
                      )

    # Make sure sum of values in gaussian kernel equals 1.
    gaussian_kernel = gaussian_kernel / torch.sum(gaussian_kernel)

    # Reshape to 2d depthwise convolutional weight
    gaussian_kernel = gaussian_kernel.view(1, 1, kernel_size, kernel_size)
    gaussian_kernel = gaussian_kernel.repeat(channels, 1, 1, 1)

    gaussian_filter = nn.Conv2d(in_channels=channels, out_channels=channels, padding=math.floor(kernel_size/2),
                                kernel_size=kernel_size, groups=channels, bias=False)

    gaussian_filter.weight.data = gaussian_kernel
    gaussian_filter.weight.requires_grad = False

    return gaussian_filter


def mask2edge(mask_img):
    with torch.no_grad():
        gf_large = get_gaussian_kernel(kernel_size=5, channels=1).to(mask_img.device)
        gf_small = get_gaussian_kernel(kernel_size=3, channels=1).to(mask_img.device)

        expand = gf_large(mask_img) > 0
        inv_expand = (~expand).float()
        shrink_outer = gf_small(inv_expand) > 0
        shrink_inner = gf_large(inv_expand) > 0
        edges = (~shrink_outer) & shrink_inner

    return edges
