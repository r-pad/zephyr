import open3d as o3d

import numpy as np
import torch
import os
import matplotlib.pyplot as plt
import cv2
import pickle
from tqdm import tqdm

from zephyr.full_pipeline.model_featurization import FeatureModel
from zephyr.full_pipeline.scene_featurization import featurizeScene
from zephyr.full_pipeline.options import getOptions, checkArgs
from zephyr.utils.bop_dataset import BopDataset
from zephyr.utils import to_np, getClosestTrans, dict_to, makeDir

from oriented_features.learned_scoring import score as generateScoreForLearning

from bop_toolkit_lib.misc import ensure_dir, depth_im_to_dist_im_fast
from oriented_features.edge_score import edgeScore

def main(args):
    dataset_root = os.path.join(args.bop_root, args.dataset_name)
    feature_sizes = [args.model_sift_feature_size]
    feature_steps = [args.model_sift_feature_size]
    dataset = BopDataset(args)
    print("Total data points:", len(dataset))

    # Get featurized model
    featured_objects = {}
    for obj_id in dataset.obj_ids:
        is_sym = obj_id in dataset.sym_obj_ids
        obj = FeatureModel(dataset_root, is_sym, args)
        obj.construct(obj_id, dataset.getObjPath(obj_id), dataset.dataset_camera)
        featured_objects[obj_id] = obj

    if args.score_data_save_dir_name is None:
        score_data_save_dir_name = "%s_%s_match_data" % (args.grid_dir_name, args.split_name)
    else:
        score_data_save_dir_name = args.score_data_save_dir_name
    save_scoring_results = args.save_scoring_results
    if save_scoring_results:
        print("Data will be saved to:", score_data_save_dir_name)

    timing_list = []
    idxs = np.arange(len(dataset))

    # Save the model info
    model_data_path = os.path.join(dataset_root, score_data_save_dir_name, "model_data")
    if not os.path.exists(model_data_path):
        os.makedirs(model_data_path)
    for obj_id in dataset.obj_ids:
        np.savez(
            os.path.join(model_data_path, "model_cloud_{:02d}.npz".format(obj_id)),
            model_points = featured_objects[obj_id].model_points,
            model_colors = featured_objects[obj_id].model_colors,
            model_normals = featured_objects[obj_id].model_normals,
        )

    for i in tqdm(idxs):
        '''Get observation data'''
        obs_data = dataset[i]
        obj_id, scene_id, im_id = obs_data['obj_id'], obs_data['scene_id'], obs_data['im_id']

        img, depth, scene_meta, depth_for_edge, mask_gt, mat_gt, scene_camera = \
            obs_data['img'], obs_data['depth'], obs_data['scene_meta'], obs_data['depth_for_edge'], \
            obs_data['mask_gt'], obs_data['mat_gt'], obs_data['scene_camera']

        sample_trans = []
        '''Get PPF results'''
        if not args.no_ppf:
            trans_ppf, ppf_scores = dataset.getPPFHypos(obj_id, scene_id, im_id)
            sample_trans.append(trans_ppf)
            n_ppf = len(trans_ppf)
        else:
            n_ppf = 0

        '''Get model data'''
        obj = featured_objects[obj_id]
        model_points, model_normals, model_colors = obj.getModelData()

        '''Featurize observation'''
        mask = np.ones_like(depth, dtype=np.uint8)
        if not args.no_sift:
            scene_cam_K = np.asarray(scene_camera['cam_K']).reshape((3,3))
            keypoints, features, cloud, frames = featurizeScene(img, depth_im_to_dist_im_fast(depth, scene_cam_K), mask, scene_meta, feature_sizes, feature_steps)

            '''Match to corresponding object'''
            trans_feat, match_aux = obj.match(features, frames, mat_gt)
            grid_idxs, obs_idxs, knn_dists = match_aux['grid_idxs'], match_aux['obs_idxs'], match_aux['knn_dists']
            target_frames = match_aux['target_frames']

            sample_trans.append(trans_feat)
            n_feat = len(trans_feat)
        else:
            cloud, target_frames = None, None
            grid_idxs, obs_idxs, knn_dists = None, None, None
            n_feat = 0

        if args.oracle_hypo:
            sample_trans.append(np.expand_dims(mat_gt, 0))

        sample_trans = np.concatenate(sample_trans, axis=0)

        pp_err = np.asarray([obj.err_func(mat[:3,:3], mat[:3, 3], mat_gt[:3, :3], mat_gt[:3, 3], model_points) for mat in sample_trans])
        ppf_scores_all = np.zeros_like(pp_err)
        ppf_scores_all[:n_ppf] = ppf_scores

        img_blur = cv2.GaussianBlur(img, (5,5), 0)

        score_res = generateScoreForLearning(
            img_blur/255., depth, cloud, target_frames, obs_idxs,
            model_points, model_colors, model_normals, None, #grid_idxs,
            sample_trans, knn_dists, scene_meta, gpu_id=args.scoring_gpu,
            verbose=False, agg_feat=False
        )

        '''For edge detection and projected area computation'''
        downsample = 5
        edge_scores, edge_scores_d, proj_area = edgeScore(img, score_res['valid_proj'], score_res['uv'], downsample, args.scoring_gpu, depth_for_edge)
        score_res['proj_area'] = proj_area
        score_res['edge_scores'] = edge_scores
        score_res['edge_scores_d'] = edge_scores_d

        score_res['depth_for_edge'] = torch.from_numpy(depth_for_edge)
        score_res['ppf_scores_all'] = ppf_scores_all
        score_res['mask'] = mask
        score_res['mask_gt'] = mask_gt

        dict_to(score_res, 'cpu')

        '''Save the results'''
        if save_scoring_results:
            del score_res['valid_depth']
            del score_res['valid_proj']
            del score_res['uv']

            score_data_save_root = os.path.join(dataset_root, score_data_save_dir_name, "%03d" % obj_id, "%04d" % scene_id)
            output_path = os.path.join(score_data_save_root, "%06d_sift_%d.npz" % (im_id, args.model_sift_feature_size))
            if not os.path.exists(os.path.dirname(output_path)):
                os.makedirs(os.path.dirname(output_path))

            np.savez(output_path,
                     cloud=cloud,
                     target_frames=target_frames,
                     obs_idxs=obs_idxs,
                     grid_idxs=grid_idxs,
                     sample_trans=sample_trans,
                     knn_dists=knn_dists,
                     meta_data=scene_meta,
                     pp_err=pp_err,
                     )

            for k, v in score_res.items():
                if type(v) == torch.Tensor and v.dtype == torch.float64:
                    score_res[k] = v.float()

            output_path = os.path.join(score_data_save_root, "%06d_score_sift_%d.pkl" % (im_id, args.model_sift_feature_size))

            with open(output_path, 'wb') as handle:
                pickle.dump(score_res, handle, protocol=pickle.HIGHEST_PROTOCOL)

if __name__ == '__main__':
    parser = getOptions()
    args = parser.parse_args()
    checkArgs(args)
    main(args)
