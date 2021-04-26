import numpy as np
from .lib import zephyr_c

COVARIANCE_MATRIX = 0
AVERAGE_3D_GRADIENT = 1
AVERAGE_DEPTH_CHANGE = 2
SIMPLE_3D_GRADIENT = 3

def compute_normals(depth,
                    fx = None,
                    fy = None,
                    cx = None,
                    cy = None,
                    depth_factor = 1.0,
                    meta_data = None,
                    method = AVERAGE_3D_GRADIENT,
                    depth_change_factor = 0.02,
                    smoothing_size = 10.0):

    if method not in [0,1,2,3]:
        raise ValueError('Invalid Method Type: {}'.format(method))
    if(meta_data is not None):
        if(fx is not None or fy is not None or \
           cx is not None or cy is not None):
            raise ValueError('Use either meta_data or [fx,fy,cx,cy]')

        fx = meta_data['camera_fx']
        fy = meta_data['camera_fy']
        cx = meta_data['camera_cx']
        cy = meta_data['camera_cy']
        depth_factor = meta_data['camera_scale']

    depth = depth.astype(np.double)
    normals = zephyr_c.compute_normals(depth, fx, fy, cx, cy,
                                                  method,
                                                  depth_change_factor,
                                                  smoothing_size,
                                                  depth_factor)

    flip_mask = (normals[:,:,2:]<0).astype(np.double) - (normals[:,:,2:]>0).astype(np.double)
    normals *= flip_mask
    return normals

def create_normal_legend(size):
    x,y = np.meshgrid(np.linspace(-1,1,size),np.linspace(-1,1,size))
    r_mask = (x*x + y*y) > 1
    x[r_mask] = np.nan
    y[r_mask] = np.nan
    z = -np.sqrt(1 - x*x - y*y)
    z[r_mask] = np.nan
    normal_legend = np.stack([x,y,z], axis=2)
    return normal_legend

def disp_normals(normals):
    disp_img = normals + np.array([[[1,1,0]]])
    disp_img = disp_img * np.array([[[.5,.5,-1]]])
    return disp_img
