'''
Run on YCB-V bop_test set in unseen mode
    python test_all.py --dataset_name ycbv
Run on YCB-V bop_test set in seen mode
    python test_all.py --dataset_name ycbv --test_seen
Run on YCB-V training set in unseen mode
    python test_all.py --dataset_name ycbv --split train

'''
import os
from pprint import pprint
import numpy as np
import matplotlib.pyplot as plt
import torch
import pickle
import cv2
import time
import random
import argparse
import pytorch_lightning as pl

from zephyr.models import getModel, PointNet2SSG
from zephyr.datasets import getDataloader
from zephyr.options import getOptions, checkArgs

from zephyr.utils.bop_dataset import BopDataset, BopDatasetArgs
from zephyr.utils.metrics import add, adi
from zephyr.utils import K2meta, to_np, meta2K, depth2cloud

from zephyr.utils.renderer import Renderer, blend
from zephyr.utils.vis import plotImages
from zephyr.utils.icp import icpRefinement
from zephyr.data_util import scanDataset
from zephyr.data_util import hypoShiftYcbv2BopBatch, modelPointsShiftYcbv2Bop

from tqdm import tqdm

from bop_toolkit_lib.visibility import estimate_visib_mask_gt
from feature_graph.utils.bop_utils import saveResultsBop

def networkInference(model, dataloader, data, point_x = None):
    if point_x is None:
        # Convert the data to pytorch data format
        scoring_data = {}
        img_blur = cv2.GaussianBlur(data['img'], (5,5), 0)
        scoring_data['img'] = torch.from_numpy(img_blur/255.)
        scoring_data['depth'] = torch.from_numpy(data['depth'])
        scoring_data['transforms'] = torch.from_numpy(data['pose_hypos'])
        scoring_data['meta_data'] = K2meta(data['cam_K'])
        scoring_data['model_points'] = torch.from_numpy(data['model_points'])
        scoring_data['model_colors'] = torch.from_numpy(data['model_colors'])
        scoring_data['model_normals'] = torch.from_numpy(data['model_normals'])
        
        # If we have GT error, store it. Otherwise use a dummy one
        if "pp_err" not in data:
            scoring_data['pp_err'] = torch.zeros(len(data['pose_hypos']))
        else:
            scoring_data['pp_err'] = data['pp_err']

        # Pre-process the data
        point_x, uv_original = dataloader.dataset.getPointNetData(scoring_data, return_uv_original = True)
    
    # Network inference
    pred_score = model({"point_x": point_x.to(model.device)})
    
    # Note that some hypotheses are excluded beforehand if it violates free-space violation too much
    # Therefore, the pp_err may be changed to a different size. 
    return pred_score, scoring_data['pp_err'], scoring_data['transforms'], uv_original

def main(main_args):
    DATASET_NAME = main_args.dataset_name
    TEST_UNSEEN = not main_args.test_seen
    RUN_HALCON_ONLINE = main_args.run_halcon_online # Whether to run the Halcon PPF on the fly
    SPLIT = main_args.split # training split is only useable for YCB-V dataset for now

    EXP_NAME = "zephyr-final"

    if DATASET_NAME == "lmo":
        PREPROCESSED_DATA_FOLDER = "./data/lmo/matches_data_test/" # The path to the preprocessed data folder
        CKPT_PATH = "./ckpts/final_lmo.ckpt" # The path to the checkpoint
        USE_ICP = False # Not using ICP for LMO dataset, as it only uses PPF hypotheses, which are already after ICP processing. 
        INCONST_RATIO_TH = 100
        BOP_ROOT = "/home/qiaog/datasets/bop/" # Change this to the root folder of downloaded BOP dataset
        PPF_RESULT_PATH = "./data/lmo_list_bop_test_v1.txt" # A text file storing all PPF pose hypotheses
        MODEL_DATA_TPATH = "./data/lmo/matches_data_test/model_data/model_cloud_{:02d}.npz" # path template to the sampled point cloud
    elif DATASET_NAME == "lm":
        PREPROCESSED_DATA_FOLDER = "./data/lmo/matches_data_test/" # The path to the preprocessed data folder
        CKPT_PATH = "./ckpts/final_lmo.ckpt" # The path to the checkpoint
        USE_ICP = False # Not using ICP for LMO dataset, as it only uses PPF hypotheses, which are already after ICP processing. 
        INCONST_RATIO_TH = 100
        BOP_ROOT = "/home/qiaog/datasets/bop/" # Change this to the root folder of downloaded BOP dataset
        PPF_RESULT_PATH = "./data/lmo_list_bop_test_v1.txt" # A text file storing all PPF pose hypotheses
        MODEL_DATA_TPATH = "./data/lmo/matches_data_test/model_data/model_cloud_{:02d}.npz" # path template to the sampled point cloud
        RUN_HALCON_ONLINE = True
    elif DATASET_NAME == "ycbv":
        if TEST_UNSEEN:
            CKPT_PATH_FOR_ODD = "./ckpts/final_ycbv_valodd.ckpt"
            CKPT_PATH_FOR_EVEN = "./ckpts/final_ycbv.ckpt"
            EXP_NAME += "-unseen"
        else:
            CKPT_PATH_FOR_ODD = "./ckpts/final_ycbv.ckpt"
            CKPT_PATH_FOR_EVEN = "./ckpts/final_ycbv_valodd.ckpt"
            EXP_NAME += "-seen"
        USE_ICP = True
        INCONST_RATIO_TH = 10
        BOP_ROOT = "/home/qiaog/datasets/bop/" # Change this to the root folder of downloaded BOP dataset
        if SPLIT == 'bop_test':
            PREPROCESSED_DATA_FOLDER = "./data/ycb/matches_data_test/" # The path to the preprocessed data folder
            PPF_RESULT_PATH = "./data/ycbv_list_bop_test.txt" # A text file storing all PPF pose hypotheses
            MODEL_DATA_TPATH = "./data/ycb/matches_data_test/model_data/model_cloud_{:02d}.npz" # path template to the sampled point cloud
        else:
            PREPROCESSED_DATA_FOLDER = "./data/ycb/matches_data_train/" # The path to the preprocessed data folder
            PPF_RESULT_PATH = "./data/ycbv_list_bop_test.txt" # A text file storing all PPF pose hypotheses
            MODEL_DATA_TPATH = "./data/ycb/matches_data_train/model_data/model_cloud_{:02d}.npz" # path template to the sampled point cloud
    else:
        raise Exception("Unknown DATASET_NAME:", DATASET_NAME)

    '''Set up the arguments for the model'''
    parser = getOptions()
    args = parser.parse_args([])

    # Model-related
    args.model_name = "pn2"
    args.dataset = "HSVD_diff_uv_norm"
    args.no_valid_proj = True
    args.no_valid_depth = True
    args.inconst_ratio_th = INCONST_RATIO_TH
    args.icp = USE_ICP

    # Dataset-related
    args.dataset_root = [PREPROCESSED_DATA_FOLDER]
    args.dataset_name = [DATASET_NAME]
    # args.resume_path = CKPT_PATH
    args.test_dataset = True

    '''Initialize pytorch dataloader and model'''
    # dataloader is only needed for the getPointNetData() function
    loader = getDataloader(args)[0]

    if DATASET_NAME == 'ycbv':
        # Load two models for ycbv. One for odd objects, another for even objects
        model = PointNet2SSG(args.dim_point, args, num_class=1)
        ckpt = torch.load(CKPT_PATH_FOR_ODD)
        model.load_state_dict(ckpt['state_dict'])
        model = model.to(0).eval()
        model_for_odd = model
        
        model = PointNet2SSG(args.dim_point, args, num_class=1)
        ckpt = torch.load(CKPT_PATH_FOR_EVEN)
        model.load_state_dict(ckpt['state_dict'])
        model = model.to(0).eval()
        model_for_even = model
    else:
        model = PointNet2SSG(args.dim_point, args, num_class=1)
        ckpt = torch.load(CKPT_PATH)
        model.load_state_dict(ckpt['state_dict'])

        model = model.to(0).eval()
    
    '''Initialize the BOP dataset'''
    # Set up the options
    if SPLIT == "bop_test":
        print("Using BOP test set")
        bop_args = BopDatasetArgs(
            bop_root=BOP_ROOT, 
            dataset_name=DATASET_NAME, 
            model_type=None,
            split_name="bop_test", # This indicates we want to use the testing set defined in BOP challenge (different than original test set)
            split="test", 
            split_type=None, 
            ppf_results_file=PPF_RESULT_PATH, 
            skip=1, # Iterate over all test samples, with no skipping
        )
    else:
        print("Using training set")
        bop_args = BopDatasetArgs(
            bop_root=BOP_ROOT, 
            dataset_name=DATASET_NAME, 
            model_type=None,
            split_name="train", # This indicates we want to use the testing set defined in BOP challenge (different than original test set)
            split="train", 
            split_type=None, 
            ppf_results_file=PPF_RESULT_PATH, 
            skip=1, # Iterate over all test samples, with no skipping
        )

    bop_dataset = BopDataset(bop_args)

    '''For testing on all LMO objects in LM objects, whether they are annotated or not'''
    all_targets = []
    if DATASET_NAME == 'lm':
        print("testing on all LMO objects in LM objects, whether they are annotated or not")
        for t in bop_dataset.targets:
            scene_id, im_id, obj_id, inst_count = t['scene_id'], t['im_id'], t['obj_id'], t['inst_count']
            if scene_id in [1, 6, 9]:
                scene_obj_ids = [1, 5, 6, 9, 10, 11, 12]
            elif scene_id in [12]:
                scene_obj_ids = [5, 6, 8, 9, 10, 11, 12]
            else:
                scene_obj_ids = [1, 5, 6, 8, 9, 10, 11, 12]
            for obj_id in scene_obj_ids:
                all_targets.append({
                    'scene_id': scene_id,
                    'im_id': im_id,
                    'obj_id': obj_id,
                    'inst_count': 1,
                })
                
        targets = random.choices(all_targets, k=6000)
        bop_dataset.targets = targets

    elif DATASET_NAME == 'ycbv' and SPLIT == 'train':
        print("FIlter out targets by pre-processed data")
        stored_datapoints = scanDataset(PREPROCESSED_DATA_FOLDER, SPLIT)
        targets_remaining = []
        for t in bop_dataset.targets:
            scene_id, im_id, obj_id, inst_count = t['scene_id'], t['im_id'], t['obj_id'], t['inst_count']
            if (obj_id, scene_id, im_id) in stored_datapoints:
                targets_remaining.append(t)
        bop_dataset.targets = targets_remaining

    print("Length of the BOP dataset:", len(bop_dataset))
    
    '''Get one datapoint in the BOP dataset'''
    BOP_RESULTS_ONLY = main_args.bop_results_only
    results = []
    renderers = {}

    for idx in tqdm(range(0, len(bop_dataset))):
        data_raw = bop_dataset[idx]

        # Extract the data needed for test forwarding
        obj_id, scene_id, im_id = data_raw['obj_id'], data_raw['scene_id'], data_raw['im_id']
        img, depth, scene_camera = data_raw['img'], data_raw['depth'], data_raw['scene_camera']
        mat_gt = data_raw['mat_gt']
        
        # The path to the full mesh model
        model_path = bop_dataset.model_tpath.format(obj_id = obj_id)
        
        cam_K = np.asarray(scene_camera['cam_K']).reshape((3, 3))

        # Load the information of the model point cloud from the pre-processed dataset
        model_data_path = MODEL_DATA_TPATH.format(obj_id)
        model_data = np.load(model_data_path)
        model_points, model_colors, model_normals = model_data['model_points'], model_data['model_colors'], model_data['model_normals']
        
        if DATASET_NAME == 'ycbv':
            # For ycbv dataset, the model points are sampled from the YCB-V dataset
            # Here shift it back to original ones
            model_points = modelPointsShiftYcbv2Bop(model_points, obj_id)
            assert model_points[:, 0].min() + model_points[:, 0].max() < 1e-2
        
        if RUN_HALCON_ONLINE:
            from zephyr.utils.halcon_wrapper import PPFModel
            
            # Create the surface model (PPF training stage)
            ppf_model = PPFModel(model_path)

            scene_pc = depth2cloud(depth, depth > 0, cam_K)
            poses_ppf, scores_ppf, time_ppf = ppf_model.find_surface_model(scene_pc * 1000.0) # The wrapper requires the input to be in milimeters
            poses_ppf[:, :3, 3] = poses_ppf[:, :3, 3] / 1000.0 # Conver from milimeter to meter
            pose_hypos = poses_ppf
        else:
            time_ppf = None
            if DATASET_NAME == "lmo":
                # Load the pose hypotheses from PPF results
                ppf_trans, ppf_scores = bop_dataset.getPPFHypos(obj_id, scene_id, im_id)
                pose_hypos = ppf_trans
            elif DATASET_NAME == "ycbv":
                # Load the all the hypotheses from prepared dataset
                prep_data = loader.dataset.loader.loadData(obj_id, scene_id, im_id)
                prep_data['transforms'] = hypoShiftYcbv2BopBatch(to_np(prep_data['transforms'][:-1]), obj_id)
                
                pose_hypos = to_np(prep_data['transforms']).copy()
        
        # Get the proper error function according to whether the object is symmetric or not
        is_sym = obj_id in bop_dataset.sym_obj_ids
        is_sym = False # Only for speed up
        err_func = adi if is_sym else add

        # Compute the ADD/ADI error for pose hypotheses w.r.t. grouth truth pose
        pp_err = np.asarray([err_func(mat[:3,:3], mat[:3, 3], mat_gt[:3, :3], mat_gt[:3, 3], model_points) for mat in pose_hypos])

        data = {
            "img": img, "depth": depth, "cam_K": cam_K, 
            "model_colors": model_colors, "model_points": model_points, "model_normals":model_normals, 
            "pose_hypos": pose_hypos
        }

        # Use the add/adi error
        data['pp_err'] = pp_err
        
        # Network inference
        t1 = time.time()
        if DATASET_NAME == 'ycbv':
            # Handle two models for YCB-V
            model = model_for_even if obj_id % 2 == 0 else model_for_odd
            pred_score, err, pose_hypos, uv_original = networkInference(model, loader, data)
        else:
            pred_score, err, pose_hypos, uv_original = networkInference(model, loader, data)
        time_zephyr = time.time() - t1
        
        pred_score = to_np(pred_score).reshape(-1)
        pose_hypos = to_np(pose_hypos)
        
        sort_idx = (-pred_score).argsort()
        pred_score = pred_score[sort_idx]
        pred_err = err[sort_idx]
        pred_pose = pose_hypos[sort_idx]
        
        # ICP
        if args.icp:
            pose_icp = pred_pose[0].copy()
            pose_icp, _ = icpRefinement(depth, to_np(uv_original[0]), pose_icp, cam_K, model_points, inpaint_depth=False, icp_max_dist=0.01)
            pred_pose[0] = pose_icp
            
        if BOP_RESULTS_ONLY:
            i = 0
            results.append({
                "obj_id": obj_id, 
                "scene_id": scene_id, 
                "im_id": im_id, 
                "err": pred_err[i], 
                "score": pred_score[i],
                "gt_pose": mat_gt,
                "pred_pose": pred_pose[i],
            })
            continue
            
        pred_score = pred_score[0]
        pred_err = pred_err[0]
        pred_pose = pred_pose[0]
        
        # Predicted pose
        # Render the object to get predicted color and depth
        if obj_id not in renderers:
            renderer = Renderer(K2meta(cam_K))
            renderer.addObject(obj_id, model_path, pose=pred_pose, mm2m=True)
            renderers[obj_id] = renderer
        else:
            renderer = renderers[obj_id]
            renderer.obj_nodes[obj_id].matrix = pred_pose
            
        pred_color, pred_depth = renderer.render()
        
        pred_mask = pred_depth > 0
        gt_mask = data_raw['mask_gt'] > 0
        gt_mask_visib = data_raw['mask_gt_visib'] > 0
        
        pred_mask_visib = estimate_visib_mask_gt(depth, pred_depth, 15/1000.)
        
        iou = np.logical_and(pred_mask, gt_mask).sum().astype(float) / np.logical_or(pred_mask, gt_mask).sum().astype(float)
        iou_visib = np.logical_and(pred_mask_visib, gt_mask_visib).sum().astype(float) / np.logical_or(pred_mask_visib, gt_mask_visib).sum().astype(float)
        
        results.append({
            "obj_id": obj_id, 
            "scene_id": scene_id, 
            "im_id": im_id, 
            "iou": iou, 
            "iou_visib": iou_visib, 
            "err": pred_err, 
            "score": pred_score.max(),
            "pred_mask_visib": pred_mask_visib,
            "pred_mask": pred_mask,
            "gt_pose": mat_gt,
            "pred_pose": pred_pose,
            "time_ppf": time_ppf,
            "time_zephyr": time_zephyr,
        })

    pickle.dump(results, open("/home/qiaog/datasets/bop/zephyr_results/%s_%s_%s_zephyr_result.pkl" % (DATASET_NAME, EXP_NAME, SPLIT), 'wb'))

    if SPLIT == 'bop_test':
        saveResultsBop(results, "/home/qiaog/results/bop_results", EXP_NAME, DATASET_NAME, pose_key='pred_pose', score_key='score', run_eval_script=True)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Arguments for this testing script')
    parser.add_argument("--dataset_name", type=str, default='lmo', choices=['lmo', 'ycbv'], help="The name of the dataset to be used")
    parser.add_argument("--run_halcon_online", action="store_true", help="If set, HALCON ppf will be run online")
    parser.add_argument("--test_seen", action="store_true", help="if set, the model trained on the same set of objects will be used. Only effective when on YCB-V")
    parser.add_argument("--bop_results_only", action="store_true", help="If set, only the pose-related results will be saved. ")
    parser.add_argument("--split", type=str, default="bop_test", choices=['bop_test', 'train'], help="The name of the dataset split the network will be tested on.")

    args = parser.parse_args()
    main(args)