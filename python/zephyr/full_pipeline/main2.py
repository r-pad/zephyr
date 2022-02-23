import open3d as o3d

from zephyr.full_pipeline.model_featurization import FeatureModel
from zephyr.full_pipeline.scene_featurization import featurizeScene
from zephyr.full_pipeline.options import getOptions
from zephyr.utils.bop_dataset import BopDataset
from zephyr.utils import to_np, getClosestTrans, dict_to

# from bop_toolkit_lib.misc import ensure_dir, depth_im_to_dist_im_fast

import numpy as np
import torch
import os
import matplotlib.pyplot as plt
import cv2
import pickle
from tqdm.notebook import tqdm
import pandas as pd
import random
import time

parser = getOptions()
args = parser.parse_args([])

args.grid_indices_path = "/home/qiaog/datasets/bop/ycbv/grid/verts_grid_0.npy"

args.grid_dir_name = "grid"
args.bop_root = "/home/qiaog/datasets/bop/"
args.dataset_name = "ycbv"
args.split = "test"
args.split_name = "bop_test"
args.ppf_results_file = "/home/qiaog/datasets/bop/ppf_data/results/ycbv_list_bop_test.txt"
args.sampled_model_dir_name = "model_pc"
args.no_sift = False
args.no_ppf = False

args.oracle_sampling = False
args.oracle_hypo = True

dataset_root = os.path.join(args.bop_root, args.dataset_name)
dataset = BopDataset(args)

feature_sizes = [args.model_sift_feature_size]
feature_steps = [args.model_sift_feature_size]

selected_obj_ids = dataset.obj_ids

featured_objects = {}
for obj_id in selected_obj_ids:
    is_sym = obj_id in dataset.sym_obj_ids
    obj = FeatureModel(dataset_root, is_sym, args, create_index=True)
    obj.construct(obj_id, dataset.getObjPath(obj_id), dataset.dataset_camera)
    featured_objects[obj_id] = obj