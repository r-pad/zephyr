import numpy as np
import os
import torch
import cv2

from oriented_features.full_pipeline.options import getOptions
from oriented_features.full_pipeline.bop_dataset import BopDataset
from oriented_features.full_pipeline.model_featurization import FeatureModel

class BopRawDataset():
    def __init__(self, bop_root, dataset_name, split, split_name, ppf_results_file, no_sift=False, no_ppf=False):
        # Get options for bop dataset
        parser = getOptions()
        args = parser.parse_args([])
        args.grid_dir_name = "grid_0.7m"
        args.dataset_name = dataset_name
        args.split = split
        args.split_name = split_name
        args.ppf_results_file = ppf_results_file
        args.no_sift = no_sift
        args.no_ppf = no_ppf
        args.bop_root = bop_root

        self.args = args
        self.dataset = BopDataset(args)
        print("BopRawDataset: len(dataset):", len(self.dataset))
        
        # Get the featurized model
        dataset_root = os.path.join(args.bop_root, args.dataset_name)
        self.featured_objects = {}
        for obj_id in self.dataset.obj_ids:
            is_sym = obj_id in self.dataset.sym_obj_ids
            obj = FeatureModel(dataset_root, is_sym, args)
            obj.construct(obj_id, self.dataset.getObjPath(obj_id), self.dataset.dataset_camera)
            self.featured_objects[obj_id] = obj

    def loadData(self, obj_id, scene_id, im_id):
        data = {}

        obs_data = self.dataset.getDataByIds(obj_id, scene_id, im_id)

        '''Get observation data'''
        img, depth, scene_meta, depth_for_edge, mask_gt, mat_gt, scene_camera = \
            obs_data['img'], obs_data['depth'], obs_data['scene_meta'], obs_data['depth_for_edge'], \
            obs_data['mask_gt'], obs_data['mat_gt'], obs_data['scene_camera']

        '''Get model data'''
        obj = self.featured_objects[obj_id]
        model_points, model_normals, model_colors = obj.getModelData()
        
        sample_trans = []
        '''Get PPF results'''
        if not self.args.no_ppf:
            trans_ppf, ppf_scores = self.dataset.getPPFHypos(obj_id, scene_id, im_id)
            sample_trans.append(trans_ppf)
            n_ppf = len(trans_ppf)
        else:
            n_ppf = 0

        assert mat_gt.shape == (4,4)
        sample_trans.append(mat_gt[None])
        sample_trans = np.concatenate(sample_trans, axis=0)
        pp_err = np.asarray([obj.err_func(mat[:3,:3], mat[:3, 3], mat_gt[:3, :3], mat_gt[:3, 3], model_points) for mat in sample_trans])
        
        img_blur = cv2.GaussianBlur(img, (5,5), 0)
        data['object_id'] = obj_id
        data['scene_id'] = scene_id
        data['im_id'] = im_id

        data['img'] = torch.from_numpy(img_blur/255.)
        data['meta_data'] = scene_meta
        data['depth'] = torch.from_numpy(depth)
        data['depth_for_edge'] = torch.from_numpy(depth_for_edge)

        data['transforms'] = torch.from_numpy(sample_trans)
        data['pp_err'] = pp_err

        data['model_points'] = torch.from_numpy(model_points)
        data['model_colors'] = torch.from_numpy(model_colors)
        data['model_normals'] = torch.from_numpy(model_normals)

        return data
