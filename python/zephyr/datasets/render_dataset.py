import os
import numpy as np

import torch
from torch.utils.data import Dataset

from zephyr.data_util import loadData, to_np, vectorize

class RenderDataset(Dataset):
    def __init__(self, datapoints, args, mode='train'):
        self.args = args
        self.datapoints = datapoints
        self.mode = mode

        self.feature_size = args.feature_size
        self.base_path = args.dataset_root
        self.max_hypos = args.max_hypos

        self.model_data_cache = {}

    def getModelData(self, obj_id):
        if obj_id not in self.model_data_cache:
            model_data_path = os.path.join(self.base_path,
                "model_data/model_cloud_{:02d}.npz".format(obj_id))
            model_data = np.load(model_data_path)
            self.model_data_cache[obj_id] = {
                "model_points": torch.from_numpy(model_data['model_points']),
                "model_colors": torch.from_numpy(model_data['model_colors']),
                "model_normals": torch.from_numpy(model_data['model_normals']),
            }

        return self.model_data_cache[obj_id]

    def __len__(self):
        return len(self.datapoints)

    def __getitem__(self, idx):
        dp = self.datapoints[idx]
        to_return = {"object_id": dp[0], "scene_id": dp[1], "im_id": dp[2]}

        '''model specific data'''
        model_data = self.getModelData(dp[0])
        to_return.update(model_data)

        '''observation specific data'''
        data = loadData(*dp, feature_size = self.feature_size, base_path = self.base_path)
        to_return['img'] = data['img'].permute(2, 0, 1)
        to_return['depth'] = data['depth'].unsqueeze(0)

        to_return['pp_err'] = data['pp_err']
        to_return['transforms'] = data['transforms']

        '''Random sampling and get max_hypos'''
        n_hypo = len(to_return['pp_err'])
        if self.mode == "train":
            idx_hypos = torch.randperm(n_hypo)
            if self.max_hypos is not None and self.max_hypos < n_hypo:
                idx_hypos = idx_hypos[:self.max_hypos]
        else:
            idx_hypos = np.arange(n_hypo)
        to_return['pp_err'] = to_return['pp_err'][idx_hypos]
        to_return['transforms'] = to_return['transforms'][idx_hypos]

        meta_data = data['meta_data'].item()
        to_return['camera_cx'] = meta_data['camera_cx']
        to_return['camera_cy'] = meta_data['camera_cy']
        to_return['camera_fx'] = meta_data['camera_fx']
        to_return['camera_fy'] = meta_data['camera_fy']

        '''Expand the meta data to match the batch size for multi-GPU compactibility'''
        for k, v in to_return.items():
            v = torch.from_numpy(np.asarray(v))
            to_return[k] = v
            if k in ['object_id', 'scene_id', 'im_id', 'camera_cx', 'camera_cy', 'camera_fx', 'camera_fy']:
                to_return[k] = v.expand(self.args.n_gpus)
            elif k in ['pp_err', 'transforms']:
                pass
            else:
                to_return[k] = v.unsqueeze(0).expand((self.args.n_gpus, *(v.shape)))
        return to_return
