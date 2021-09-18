import os

import sys
sys.path.insert(0, "/home/qiaog/src/pyfqmr-Fast-Quadric-Mesh-Reduction")

import cv2

import math
import numpy as np

os.environ['PYOPENGL_PLATFORM'] = 'egl'
import trimesh
import pyrender

class Renderer():
    def __init__(self, meta):
        self.scene = pyrender.Scene(ambient_light=np.ones(3))
        camera = pyrender.IntrinsicsCamera(fx=meta['camera_fx'],
                                            fy=meta['camera_fy'],
                                            cx=meta['camera_cx'],
                                            cy=meta['camera_cy'])
        camera_pose = quaternion_matrix([1,0,0,0])
        self.scene.add(camera, pose=camera_pose)

        self.obj_nodes = {}

    def addObject(self, obj_id, obj_filename, pose = np.eye(4), mm2m = False, simplify=False):
        assert obj_id not in self.obj_nodes
        obj_trimesh = trimesh.load(obj_filename)

        # if simplify:
        #     from pySimplify import pySimplify
        #     simplify = pySimplify()
        #     simplify.setMesh(obj_trimesh)
        #     simplify.simplify_mesh(target_count = 1024, aggressiveness=7, preserve_border=True, verbose=10)
        #     obj_trimesh = simplify.getMesh()

        if mm2m:
            obj_trimesh.units = 'mm'
            obj_trimesh.convert_units('m')
        obj_mesh = pyrender.Mesh.from_trimesh(obj_trimesh)
        obj_node = self.scene.add(obj_mesh, pose=pose)

        self.obj_nodes[obj_id] = obj_node
    
    def render(self, w=640, h=480, depth_only=False):
        renderer = pyrender.OffscreenRenderer(viewport_width=w,
                                                viewport_height=h,
                                                point_size=1.0)
        if depth_only:
            rend_depth = renderer.render(self.scene, pyrender.RenderFlags.DEPTH_ONLY)
            rend_color = None
        else:
            rend_color, rend_depth = renderer.render(self.scene)
        return rend_color, rend_depth
    
def increase_brightness(img, value=30):
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv)

    lim = 255 - value
    v[v > lim] = 255
    v[v <= lim] += value

    final_hsv = cv2.merge((h, s, v))
    img = cv2.cvtColor(final_hsv, cv2.COLOR_HSV2BGR)
    return img

def blend(img, rend_color, rend_depth):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    rend_mask = np.expand_dims((rend_depth > 0).astype(float), -1)
    pure_white = np.ones_like(gray) * 255
    gray = (pure_white.astype(float) * 0.5 + gray.astype(float) * (1-0.5)).astype(np.uint8)
    gray_blend = np.expand_dims(gray, -1)

    blend = rend_color.astype(float) * rend_mask + gray_blend.astype(float) * (1 - rend_mask)
    blend = blend.round().astype(np.uint8)
    return blend

# epsilon for testing whether a number is close to zero
_EPS = np.finfo(float).eps * 4.0

def quaternion_matrix(quaternion):
    """Return homogeneous rotation matrix from quaternion.

    >>> R = quaternion_matrix([0.06146124, 0, 0, 0.99810947])
    >>> np.allclose(R, rotation_matrix(0.123, (1, 0, 0)))
    True

    """
    q = np.array(quaternion[:4], dtype=np.float64, copy=True)
    nq = np.dot(q, q)
    if nq < _EPS:
        return np.identity(4)
    q *= math.sqrt(2.0 / nq)
    q = np.outer(q, q)
    return np.array((
        (1.0-q[1, 1]-q[2, 2],     q[0, 1]-q[2, 3],     q[0, 2]+q[1, 3], 0.0),
        (    q[0, 1]+q[2, 3], 1.0-q[0, 0]-q[2, 2],     q[1, 2]-q[0, 3], 0.0),
        (    q[0, 2]-q[1, 3],     q[1, 2]+q[0, 3], 1.0-q[0, 0]-q[1, 1], 0.0),
        (                0.0,                 0.0,                 0.0, 1.0)
        ), dtype=np.float64)