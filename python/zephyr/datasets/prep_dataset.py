import os
import numpy as np
import torch

from zephyr.data_util import loadData


class PrepDataset():
    def __init__(self, dataset_root, feature_size):
        self.dataset_root = dataset_root
        self.feature_size = feature_size
        self.model_data_cache = {}

    def loadData(self, obj_id, scene_id, im_id):
        data = loadData(obj_id, scene_id, im_id, feature_size = self.feature_size, base_path = self.dataset_root)

        '''Get the model data and send it into the processing function'''
        model_data = self.getModelData(obj_id)
        data.update(model_data)

        return data

    def getModelData(self, obj_id):
        if obj_id not in self.model_data_cache:
            model_data_path = os.path.join(self.dataset_root,
                "model_data/model_cloud_{:02d}.npz".format(obj_id))
            model_data = np.load(model_data_path)
            self.model_data_cache[obj_id] = {
                "model_points": torch.from_numpy(model_data['model_points']),
                "model_colors": torch.from_numpy(model_data['model_colors']),
                "model_normals": torch.from_numpy(model_data['model_normals']),
            }

        return self.model_data_cache[obj_id]