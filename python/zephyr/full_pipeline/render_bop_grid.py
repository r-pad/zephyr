import os
import cv2
import argparse
import pathlib
import json
import numpy as np
from tqdm import tqdm
import scipy.io as scio
from PIL import Image
from glob import glob
from functools import partial

from object_pose_utils.bbTrans.discretized4dSphere import S3Grid

from scipy.spatial.transform import Rotation as R
from zephyr.utils.renderer import Renderer

from multiprocessing import Pool

def is_int(s):
    try: 
        int(s)
        return True
    except ValueError:
        return False

def quaternion_matrix(quaternion):
    return R.from_quat(quaternion).as_matrix()

def getTransform(q, t=[0,0,.5]):
    trans_mat = np.eye(4)
    trans_mat[:3, :3] = quaternion_matrix(q)
    trans_mat[:3,3] = t
    return trans_mat

def renderOneObject(model_filename, quats, digits, camera_data):
    (model, ext) = os.path.splitext(os.path.basename(model_filename))

    camera_data['camera_fx'] = camera_data['fx']
    camera_data['camera_fy'] = camera_data['fy']
    camera_data['camera_cx'] = camera_data['cx']
    camera_data['camera_cy'] = camera_data['cy']
    
    renderer = Renderer(camera_data)
    
    renderer.addObject(0, model_filename, mm2m=True)
    
    render_dir = os.path.join(args.output_dir, model)
    pathlib.Path(render_dir).mkdir(parents=True, exist_ok=True)
    filename_template = render_dir + '/{:0'+str(digits)+'d}-{}.{}'
    
    for j, q in tqdm(enumerate(quats), total=len(quats)):
        obj_label = 1
        t = [0,0,args.camera_distance]
        trans_mat = getTransform(q, t)
        renderer.obj_nodes[0].matrix = trans_mat
        img, depth = renderer.render()
        
        # Convert image from RGB to BGR for OpenCV convension
        img = img[:, :, ::-1]
        
        depth[depth > 1000] = 0
        depth = depth*10000
        cv2.imwrite(filename_template.format(j, 'color', 'png'), img)
        Image.fromarray(depth.astype(np.int32), "I").save(filename_template.format(j,'depth', 'png'))
        np.save(filename_template.format(j, 'trans', 'npy'), q)
        label = np.where(np.array(img[:,:,-1])==255, obj_label, 0)
        cv2.imwrite(filename_template.format(j,'label', 'png'), label)
        poses = np.zeros([3,4,1])
        poses[:3,:3,0] = quaternion_matrix(q)[:3,:3]
        poses[:3,3,0] = t
        scio.savemat(filename_template.format(j,'meta', 'mat'), 
                    {'cls_indexes':np.array([[obj_label]]), 
                    'factor_depth':np.array([[10000]]),
                    'poses':poses,
                    'camera':camera_data})


def main(args):
    camera_json = glob(os.path.join(args.dataset_root, 'camera*.json'))[0]

    if(args.output_dir is None):
        args.output_dir = os.path.join(args.dataset_root, args.quats)

    with open(camera_json) as f:
        camera_data=camera_data = json.load(f)

    if 'tless' in args.dataset_root:
        with open(os.path.join(args.dataset_root, 'models_cad', 'models_info.json')) as f:
            model_info = json.load(f) 
        ply_files = glob(os.path.join(args.dataset_root, 'models_cad', '*.ply'))
        fix_ply_color=True
    else:
        with open(os.path.join(args.dataset_root, 'models_fine', 'models_info.json')) as f:
            model_info = json.load(f) 
        ply_files = glob(os.path.join(args.dataset_root, 'models_fine', '*.ply'))
        fix_ply_color=False
    ply_files.sort()

    if(args.quats == 'grid'):
        grid = S3Grid(2)
        grid.Simplify()
        quats = grid.vertices
    elif(is_int(args.quats)):
        quats = [random_quaternion() for _ in range(int(args.quats))]
    elif(args.quats[-4:] == '.txt'):
        with open(args.quats) as f:
            quats = f.readlines()
        for j, q in enumerate(quats):
            quats[j] = np.array(q, dtype=float)
            quats[j] /= np.linalg.norm(quats[j])
    else:
        raise ValueError('Bad quaternion format. Valid formats are \'grid\', N, or [file].txt')

    digits = len(str(len(quats)))

    
    with Pool(12) as p:
        p.map(partial(renderOneObject, quats=quats, digits=digits, camera_data=camera_data), ply_files)

    return

    pbar_model = tqdm(ply_files)
    for model_filename in pbar_model:
        (model, ext) = os.path.splitext(os.path.basename(model_filename))

        pbar_model.set_description('Rendering {}'.format(model))
        
        camera_data['camera_fx'] = camera_data['fx']
        camera_data['camera_fy'] = camera_data['fy']
        camera_data['camera_cx'] = camera_data['cx']
        camera_data['camera_cy'] = camera_data['cy']
        
        renderer = Renderer(camera_data)
        
        renderer.addObject(0, model_filename, mm2m=True)
        
        render_dir = os.path.join(args.output_dir, model)
        pathlib.Path(render_dir).mkdir(parents=True, exist_ok=True)
        filename_template = render_dir + '/{:0'+str(digits)+'d}-{}.{}'
        
        for j, q in tqdm(enumerate(quats), total=len(quats)):
            obj_label = 1
            t = [0,0,args.camera_distance]
            trans_mat = getTransform(q, t)
            renderer.obj_nodes[0].matrix = trans_mat
            img, depth = renderer.render()
            
            # Convert image from RGB to BGR for OpenCV convension
            img = img[:, :, ::-1]
            
            depth[depth > 1000] = 0
            depth = depth*10000
            cv2.imwrite(filename_template.format(j, 'color', 'png'), img)
            Image.fromarray(depth.astype(np.int32), "I").save(filename_template.format(j,'depth', 'png'))
            np.save(filename_template.format(j, 'trans', 'npy'), q)
            label = np.where(np.array(img[:,:,-1])==255, obj_label, 0)
            cv2.imwrite(filename_template.format(j,'label', 'png'), label)
            poses = np.zeros([3,4,1])
            poses[:3,:3,0] = quaternion_matrix(q)[:3,:3]
            poses[:3,3,0] = t
            scio.savemat(filename_template.format(j,'meta', 'mat'), 
                        {'cls_indexes':np.array([[obj_label]]), 
                        'factor_depth':np.array([[10000]]),
                        'poses':poses,
                        'camera':camera_data})


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_root', type=str, default =  '/datasets/bop/tless', 
        help='model root dir')
    parser.add_argument('--output_dir', type=str, default = None,
        help='Data render output location')
    parser.add_argument('--quats', type=str, default = 'grid', help='\'grid\' for 3885 grid or N for random quaternions')
    parser.add_argument('--camera_distance', type=float, default = 1, 
        help='camera distance')

    args = parser.parse_args()
    main(args)