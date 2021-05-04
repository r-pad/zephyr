import numpy as np
import pandas as pd
import imageio
import os, sys
import json
import glob
from scipy.spatial.transform import Rotation as R

from dataclasses import dataclass

# The following line append the folder of the bop_toolkit to the system path
# And it enables importing bop_toolkit_lib
sys.path.append(os.path.join(os.path.dirname(__file__), "../../../external/bop_toolkit/"))

from bop_toolkit_lib.dataset_params import get_model_params, get_camera_params, get_split_params
from bop_toolkit_lib.inout import load_im, load_depth, load_json

@dataclass
class BopDatasetArgs:
    bop_root: str
    dataset_name: str
    model_type: str # Mostly None, see get_model_params()
    split_name: str # "bop_test" and "test" will be treated differently, otherwise, same as split
    split: str # ('train', 'val', 'test')
    split_type: str # Mostly None, see get_split_params()
    ppf_results_file: str # The path to the pre-generted PPF pose estimation results
    skip: int = 50 # Sample only one frames in consective skip frames. Ignored when using bop_test split

class BopDataset():
    '''
    This is a wrapper class for Loading BOP datasets in bop_toolkit
    It returns various data needed to train or test the network
    '''
    def __init__(self, args):
        self.bop_root = args.bop_root
        self.dataset_name = args.dataset_name
        self.dataset_root = os.path.join(self.bop_root, self.dataset_name)
        self.model_type = args.model_type
        self.split_name = args.split_name # "bop_test" and "test" will be treated differently
        self.split = args.split # ('train', 'val', 'test')
        self.split_type = args.split_type # (e.g. for T-LESS, possible types of the 'train' split are: 'primesense', 'render_reconst')

        ppf_resuls_root = os.path.join(self.bop_root, "ppf_data", "results")

        # Get the information using bop_toolkit_lib
        model_params = get_model_params(self.bop_root, self.dataset_name, self.model_type)
        self.dataset_camera = get_camera_params(self.bop_root, self.dataset_name, None)

        self.obj_ids = model_params['obj_ids']
        self.sym_obj_ids = model_params['symmetric_obj_ids']
        self.model_tpath = model_params['model_tpath']
        self.models_info_path = model_params['models_info_path']

        # Get the distribution of this split and the test split
        test_split_params = get_split_params(self.bop_root, self.dataset_name, "test")
        test_depth_range = test_split_params['depth_range']
        self.depth_clip_up = (test_depth_range[1] + 200) / 1000.0 # in meter; The upper bound of the depth of test images + 20 cm

        self.split_params = get_split_params(self.bop_root, self.dataset_name, self.split, self.split_type)
        self.split_dir_name = self.split_params['split_path'].split("/")[-1]

        uois_tpath = self.split_params['rgb_tpath']
        self.uois_tpath = uois_tpath.replace("rgb", "uois").replace("png", "npz")

        # Scan the dataset and get the dataset split we want
        self.targets = getTargets(self.dataset_root, self.split_name, self.split_dir_name, skip=args.skip)

        # Load PPF results
        if args.ppf_results_file is None:
            self.ppf_results_file = os.path.join(ppf_resuls_root, "%s_list_%s.txt" % (self.dataset_name, self.split_name))
        else:
            self.ppf_results_file = args.ppf_results_file
            
        self.df_ppf = None

        self.scene_gt_info_cache = {}
        self.scene_gt_cache = {}

    def getSceneGtInfo(self, scene_id):
        if scene_id in self.scene_gt_info_cache:
            return self.scene_gt_info_cache[scene_id]
        else:
            scene_gt_info = load_json(self.split_params['scene_gt_info_tpath'].format(scene_id=scene_id))
            self.scene_gt_info_cache[scene_id] = scene_gt_info
            return self.scene_gt_info_cache[scene_id]

    def getSceneGt(self, scene_id):
        if scene_id in self.scene_gt_cache:
            return self.scene_gt_cache[scene_id]
        else:
            scene_gt = load_json(self.split_params['scene_gt_tpath'].format(scene_id=scene_id))
            self.scene_gt_cache[scene_id] = scene_gt
            return self.scene_gt_cache[scene_id]

    def getObjPath(self, obj_id):
        return self.model_tpath.format(obj_id=obj_id)

    def getMetaDataByIds(self, obj_id, scene_id, im_id):
        scene_gt_info = self.getSceneGtInfo(scene_id)[str(im_id)]
        scene_gt = self.getSceneGt(scene_id)[str(im_id)]

        for gt_id, scene_obj_gt in enumerate(scene_gt):
            if scene_obj_gt['obj_id'] == obj_id:
                break
            if gt_id == len(scene_gt)-1:
                raise Exception("Incorrect ground truth")

        visib_fract = scene_gt_info[gt_id]['visib_fract']

        data = {
            "visib_fract": visib_fract
        }

        return data

    def getDataByIds(self, obj_id, scene_id, im_id):
        scene_gt_info = load_json(self.split_params['scene_gt_info_tpath'].format(scene_id=scene_id))[str(im_id)]
        scene_camera = load_json(self.split_params['scene_camera_tpath'].format(scene_id=scene_id))[str(im_id)]
        scene_gt = load_json(self.split_params['scene_gt_tpath'].format(scene_id=scene_id))[str(im_id)]

        for gt_id, scene_obj_gt in enumerate(scene_gt):
            if scene_obj_gt['obj_id'] == obj_id:
                break
            if gt_id == len(scene_gt)-1:
                raise Exception("Incorrect ground truth")
            
        visib_fract = scene_gt_info[gt_id]['visib_fract']

        cam_R_m2c = np.asarray(scene_obj_gt['cam_R_m2c']).reshape((3,3))
        cam_t_m2c = np.asarray(scene_obj_gt['cam_t_m2c']).reshape((3,1))/1000.0 # convert it to meters
        mat_gt = np.hstack((cam_R_m2c, cam_t_m2c))
        mat_gt = np.vstack((mat_gt, np.asarray([0,0,0,1])))

        img = load_im(self.split_params['rgb_tpath'].format(scene_id=scene_id, im_id=im_id))
        depth = load_depth(self.split_params['depth_tpath'].format(scene_id=scene_id, im_id=im_id))
        mask_gt = imageio.imread(self.split_params['mask_tpath'].format(scene_id=scene_id, im_id=im_id, gt_id=gt_id))
        mask_gt_visib = imageio.imread(self.split_params['mask_visib_tpath'].format(scene_id=scene_id, im_id=im_id, gt_id=gt_id))

        '''Convert the depth to meters'''
        depth = depth * scene_camera['depth_scale'] / 1000.0

        scene_meta = {'object_label': obj_id, "im_id": im_id, "scene_id": scene_id}
        # Dividing the depth by this number converts the depth from milimeter to meter
        # scene_meta['camera_scale'] = 1000.0 / scene_camera['depth_scale']
        scene_meta['camera_scale'] = 1.0
        scene_meta['camera_fx'] = scene_camera['cam_K'][0]
        scene_meta['camera_cx'] = scene_camera['cam_K'][2]
        scene_meta['camera_fy'] = scene_camera['cam_K'][4]
        scene_meta['camera_cy'] = scene_camera['cam_K'][5]

        scene_cam_K = np.asarray(scene_camera['cam_K']).reshape((3,3))

        depth_for_edge = depth.clip(0, self.depth_clip_up)

        data = {
            "img": np.asarray(img),
            "depth": np.asarray(depth),
            "mat_gt": mat_gt,
            "mask_gt": np.asarray(mask_gt),
            "mask_gt_visib": np.asarray(mask_gt_visib),
            "scene_meta": scene_meta,
            "scene_camera": scene_camera,
            "depth_for_edge": depth_for_edge,
            "obj_id": obj_id,
            "scene_id": scene_id,
            "im_id": im_id,
            "visib_fract": visib_fract,
        }
        return data

    def getPPFHypos(self, obj_id, scene_id, im_id):
        if self.df_ppf is None:
            self.df_ppf = pd.read_csv(self.ppf_results_file, delimiter='\t')

        ppf_results = getHalconPPFResults(self.df_ppf, obj_id, scene_id, im_id)
        ppf_trans = np.asarray([_[0] for _ in ppf_results])
        ppf_scores = np.asarray([_[1] for _ in ppf_results])
        return ppf_trans, ppf_scores

    def __getitem__(self, idx):
        t = self.targets[idx]
        obj_id = t['obj_id']
        scene_id = t['scene_id']
        im_id = t['im_id']
        inst_count = t['inst_count']

        if inst_count > 1:
            print("inst_count > 1:", self.dataset_name, obj_id, scene_id, im_id)

        data = self.getDataByIds(obj_id, scene_id, im_id)
        data['obj_id'] = obj_id
        data['scene_id'] = scene_id
        data['im_id'] = im_id

        return data

    def __len__(self):
        return len(self.targets)



def getTargets(dataset_root, split_name, split_dir_name, skip=1):
    if split_name == "bop_test":
        targets = json.load(open(os.path.join(dataset_root, "test_targets_bop19.json")))
    else:
        targets = []
        split_root = os.path.join(dataset_root, split_dir_name)
        scenes = glob.glob(os.path.join(split_root, "*/"))
        scenes.sort()
        scene_ids = [int(_.split("/")[-2]) for _ in scenes]

        for scene_id in scene_ids:
            scene_root = os.path.join(split_root, "%06d" % scene_id)
            scene_gt = json.load(open(os.path.join(scene_root, "scene_gt.json")))
            im_ids = [int(_) for _ in scene_gt.keys()]

            im_ids = im_ids[::skip]

            if len(set(im_ids)) != len(im_ids):
                print("Warning: multiple instance!")

            for im_id in im_ids:
                im_gt = scene_gt[str(im_id)]
                for t in im_gt:
                    obj_id = t['obj_id']
                    targets.append({
                        'im_id': im_id,
                        "scene_id": scene_id,
                        "obj_id": obj_id,
                        "inst_count": 1
                    })
    return targets

def getHalconPPFResults(df, object_id, scene_id, im_id, get_n=1e6, scale=1/1000.0):
    df_this_result = df[(df['ObjectId'] == object_id) & (df['SceneId'] == scene_id) & (df['ImageId'] == im_id)]
    df_this_result = df_this_result.sort_values('Score', ascending=False)
    results = []
    for score_rank in range(min(get_n, len(df_this_result))):
        res = df_this_result.iloc[score_rank]

        rot = R.from_euler("XYZ", [res['XRot'], res['YRot'], res['ZRot']], degrees=True)
        rotmat = rot.as_matrix()
        mat_est = np.eye(4)
        mat_est[:3, :3] = rotmat
        mat_est[:3, 3] = np.array([res['XTrans'], res['YTrans'], res['ZTrans']]) * scale # converts milimeter to meters
        score = res['Score']
        results.append((mat_est, score))
    return results