import cv2
import torch
import torch.nn as nn
import numpy as np
import os, pickle, glob
import csv
import copy

from oriented_features.pose_scoring_lightning.constants import YCBV_TRAIN_SCENE, YCBV_VALID_SCENE, YCBV_BOPTEST_SCENE

def loadData(object_id, scene_id, im_id, feature_size = 11, base_path = "/datasets/ycb/matches_data_full/"):
    try:
        score_sift_path = os.path.join(base_path, "%03d/%04d/%06d_scores_sift_%d.pkl" % (object_id, scene_id, im_id, feature_size))
        score_data = pickle.load(open(score_sift_path, "rb"))
    except:
        score_sift_path = os.path.join(base_path, "%03d/%04d/%06d_score_sift_%d.pkl" % (object_id, scene_id, im_id, feature_size))
        score_data = pickle.load(open(score_sift_path, "rb"))
    sift_path = os.path.join(base_path, "%03d/%04d/%06d_sift_%d.npz" % (object_id, scene_id, im_id, feature_size))
    if os.path.exists(sift_path):
        sift_data_npz = np.load(sift_path, allow_pickle=True)
        sift_data = dict()
        for k in sift_data_npz.files:
            sift_data[k] = sift_data_npz[k]
        score_data.update(sift_data)

    '''Try to load the uois mask'''
    uois_path = os.path.join(base_path, "*/%04d/%06d_uois.npz" % (scene_id, im_id))
    uois_path = glob.glob(uois_path)
    if len(uois_path) >= 1:
        uois_path = uois_path[0]
        uois_data = np.load(uois_path)
        # print(uois_data.files)
        score_data['uois_mask'] = uois_data['seg_masks']

    if 'meta_data' in score_data:
        if type(score_data['meta_data']) is dict:
            pass
        else:
            score_data['meta_data'] = score_data['meta_data'].item()

    if "transforms" in score_data:
        score_data['transforms'][:, 3, 0:3] = 0

    return score_data

def scanDataset(base_path = "/datasets/ycb/matches_data/", split="valid"):
    train_scenes = YCBV_TRAIN_SCENE
    valid_scenes = YCBV_VALID_SCENE
    boptest_scenes = YCBV_BOPTEST_SCENE

    paths = glob.glob(os.path.join(base_path, "*/*/*"))
    paths.sort()
    datapoints = []
    for p in paths:
        parts = p.split("/")[-3:]
        if not parts[0].isdigit():
            continue
        oid = int(parts[0])
        sid = int(parts[1])
        iid = int(parts[2].split("_")[0])
        if (oid, sid, iid) in datapoints:
            continue
        else:
            datapoints.append((oid, sid, iid))
    if split == "valid":
        datapoints = [_ for _ in datapoints if _[1] in valid_scenes]
    elif split == "train":
        datapoints = [_ for _ in datapoints if _[1] in train_scenes]
    elif split == "bop_test":
        datapoints = [_ for _ in datapoints if _[1] in boptest_scenes]
    elif split == "all":
        pass
    else:
        raise Exception("Unknown split:", split)

    return datapoints

def img2uint8(img):
    if img.dtype == np.uint8:
        return img
    else:
        if img.max() <= 1:
            return (img * 255).round().clip(0,255).astype(np.uint8)
        else:
            return img.round().clip(0,255).astype(np.uint8)

def to_np(x):
    if torch.is_tensor(x):
        return x.detach().cpu().numpy()
    else:
        return x.detach().data.cpu().numpy()

def vectorize(x, use_hsv=True, feature_inliers=True, norm_cos_weight=False, fs_thresh=0.02):
    score_vec = x['score_vec']

    edge_scores = x['edge_scores']
    edge_scores_d = x['edge_scores_d']

    score_vec = torch.cat([
        score_vec,
        edge_scores.float().unsqueeze(-1),
        edge_scores_d.float().unsqueeze(-1)
    ], axis=-1)

    if use_hsv:
        h_err = x['h_err']
        s_err = x['s_err']
        v_err = x['v_err']

        score_vec = torch.cat([
            score_vec,
            h_err.unsqueeze(-1).float(),
            s_err.unsqueeze(-1).float(),
            v_err.unsqueeze(-1).float(),
        ], axis=-1)

    '''For feature inliers'''
    if feature_inliers:
        inlier_th = 0.02
        if ("inlier_feature_dists" not in x) and ("obs_feature_match_count" not in x):
            assert "corre_dists" in x
            unique_corre = x['unique_corre']
            unique_feat_dist = x['unique_feat_dist']
            corre_dists = x['corre_dists']
            N = corre_dists.shape[0]

            # Compute the Euclidean and feature distances for the projected feature inliers
            corre_inlier_mask = corre_dists < inlier_th
            inlier_feature_dists = []
            inlier_euclidean_dists = []
            inlier_idxs = []
            for i in range(N):
                inlier_feature_dists.append(unique_feat_dist[corre_inlier_mask[i]])
                inlier_euclidean_dists.append(corre_dists[i, corre_inlier_mask[i]])
                inlier_idxs.append(unique_corre[:, corre_inlier_mask[i]])

            x['inlier_feature_dists'] = inlier_feature_dists
            x['inlier_euclidean_dists'] = inlier_euclidean_dists
            x['inlier_idxs'] = inlier_idxs

        if 'obs_feature_match_count' not in x:
            inlier_feature_dists = x['inlier_feature_dists']
            inlier_euclidean_dists = x['inlier_euclidean_dists']
            inlier_idxs = x['inlier_idxs']

            inlier_count = torch.tensor([len(_) for _ in inlier_feature_dists], device=color_err.device).float()
            obs_feature_match_count = torch.tensor([len(np.unique(_[0])) for _ in inlier_idxs], device=color_err.device).float()
            obs_feature_match_ratio = obs_feature_match_count / obs_feature_match_count.max() # We didn't record the number of obs features
            inlier_feature_dist_sum = torch.tensor([sum(_) for _ in inlier_feature_dists], device=color_err.device).float()
            inlier_feature_dist_mean = inlier_feature_dist_sum / inlier_count
            inlier_euclidean_dist_sum = torch.tensor([sum(_) for _ in inlier_euclidean_dists], device=color_err.device).float()
            inlier_euclidean_dist_mean = inlier_euclidean_dist_sum / inlier_count
        else:
            inlier_count = x['inlier_count'].float()
            obs_feature_match_count = x['obs_feature_match_count'].float()
            # obs_feature_match_ratio = x['obs_feature_match_ratio'].float()
            proj_area = x['proj_area'].float().squeeze()
            obs_feature_match_ratio = obs_feature_match_count / (proj_area + 1)
            inlier_feature_dist_sum = x['inlier_feature_dist_sum'].float()
            inlier_feature_dist_mean = x['inlier_feature_dist_mean'].float()
            inlier_euclidean_dist_sum = x['inlier_euclidean_dist_sum'].float()
            inlier_euclidean_dist_mean = x['inlier_euclidean_dist_mean'].float()

        score_vec = torch.cat([
            score_vec,
            inlier_count.unsqueeze(-1),
            obs_feature_match_count.unsqueeze(-1),
            obs_feature_match_ratio.unsqueeze(-1),
            inlier_feature_dist_sum.unsqueeze(-1),
            inlier_feature_dist_mean.unsqueeze(-1),
            inlier_euclidean_dist_sum.unsqueeze(-1),
            inlier_euclidean_dist_mean.unsqueeze(-1),
        ], axis=-1)

    # eliminate NaN in the input vectors
    score_vec[score_vec != score_vec] = 0

    return score_vec

def dict_cat(dict_base, dict_new, axis = 0, no_cat = ['unique_corre', 'unique_feat_dist']):
    for k,v in dict_new.items():
        if(k not in dict_base.keys()):
            dict_base[k] = v
        elif(k not in no_cat):
            if(type(v) is torch.Tensor):
                dict_base[k]=torch.cat([dict_base[k], v], axis=axis)
            elif(type(v) is np.ndarray):
                dict_base[k]=np.concatenate([dict_base[k], v], axis=axis)

def dict_to(dictionary, device):
    for k,v in dictionary.items():
        if(type(v) is torch.Tensor):
            dictionary[k]=v.to(device)

def histogramEachRow(data_in, bins = 100):
    if type(data_in) is torch.Tensor:
        data = to_np(data_in)
    else:
        data = data_in

    results = []
    for row in data:
        hist, bin_edges = np.histogram(row, bins = bins)
        results.append(np.concatenate([hist, bin_edges]))
    results = np.asarray(results)

    if type(data_in) is torch.Tensor:
        results = torch.from_numpy(results).to(data_in.device)
    return results

'''Input is a (N, 2*M+1) matrix. Each row represent the count and the bin edges'''
def histogramFilterCount(data, func):
    if len(data.shape) < 2:
        data.reshape((1, -1))
    N = data.shape[0]
    M = data.shape[1] // 2
    count = data[:, :M]
    edges = data[:, M:]
    good_edges = func(edges)
    good_bins = (good_edges[:, 1:] + good_edges[:, :-1]) > 0
    return (count * good_bins).sum(1)

def histogramFilterSum(data, func):
    if len(data.shape) < 2:
        data.reshape((1, -1))
    N = data.shape[0]
    M = data.shape[1] // 2
    count = data[:, :M]
    edges = data[:, M:]
    bin_middle = (edges[:, 1:] + edges[:, :-1]) / 2
    good_edges = func(edges)
    good_bins = (good_edges[:, 1:] + good_edges[:, :-1]) > 0
    return (count * good_bins * bin_middle).sum(1)

def histogramMean(data):
    if len(data.shape) < 2:
        data.reshape((1, -1))
    N = data.shape[0]
    M = data.shape[1] // 2
    count = data[:, :M]
    edges = data[:, M:]
    bin_middle = (edges[:, 1:] + edges[:, :-1]) / 2
    row_sum = (bin_middle * count).sum(1)
    row_count = count.sum(1)
    return row_sum / row_count

def histogramSum(data):
    if len(data.shape) < 2:
        data.reshape((1, -1))
    N = data.shape[0]
    M = data.shape[1] // 2
    count = data[:, :M]
    edges = data[:, M:]
    bin_middle = (edges[:, 1:] + edges[:, :-1]) / 2
    row_sum = (bin_middle * count).sum(1)
    return row_sum

def saveOutputs(outputs, save_path):
    print("Saving test logs to", save_path)
    with open(save_path, 'w', newline='') as csvfile:
        fieldnames = outputs[0].keys()
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames, quoting = csv.QUOTE_NONE)

        writer.writeheader()
        for i in range(len(outputs)):
            o = copy.deepcopy(outputs[i])

            for k, v in o.items():
                if type(v) is torch.Tensor:
                    o[k] = to_np(v)

            for k in fieldnames:
                if k.startswith("mat"):
                    o[k] = " ".join(["%0.6f"%_ for _ in o[k].reshape(-1).tolist()])
            writer.writerow(o)

def convertResultsBOP(input_root, output_root, input_file, dataset_name, subname=None, mat_name="mat", ycbv_shift=True):
    import pandas as pd
    input_name = input_file[:-4]
    input_name = input_name.replace("_", "-")
    input_name = input_name.replace("=", "-")

    input_path = os.path.join(input_root, input_file)

    if subname is None:
        output_path = os.path.join(output_root, "%s_%s-test.csv" % (input_name, dataset_name))
    else:
        output_path = os.path.join(output_root, "%s-%s_%s-test.csv" % (input_name, subname, dataset_name))

    df = pd.read_csv(input_path)

    csv_file = open(output_path, mode="w")
    fieldnames = ["scene_id","im_id","obj_id","score","R","t","time"]
    csv_writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
    csv_writer.writeheader()

    for i, r in enumerate(df.iterrows()):
        data = r[1]
        object_id = data['object_id']
        scene_id = data['scene_id']
        im_id = data['im_id']

        mat = data[mat_name]
        mat = [float(_) for _ in mat.split(" ")]
        mat = np.asarray(mat).reshape((4,4))

        if dataset_name == "ycbv" and ycbv_shift:
            modelShiftBopYcbv(mat, object_id)
        else:
            mat[:3, 3] = mat[:3, 3] * 1000.0

        # mat[:3, 3] = mat[:3, 3] * 1000.0

        csv_writer.writerow({
            "scene_id": scene_id,
            "im_id": im_id,
            "obj_id": object_id,
            "score": 1,
            "R": " ".join([str(_) for _  in mat[:3, :3].flatten()]),
            "t": " ".join([str(_) for _  in mat[:3, 3].flatten()]),
            "time": -1,
        })

    print("BOP logs saved to:", output_path)
    csv_file.close()

    return


# in milimeter
YCBV_MODEL_SHIFTS = {
    1:[1.3360, -0.5000, 3.5105],
    2:[0.5575, 1.7005, 4.8050],
    3:[-0.9520, 1.4670, 4.3645],
    4:[-0.0240, -1.5270, 8.4035],
    5:[1.2995, 2.4870, -11.8290],
    6:[-0.1565, 0.1150, 4.2625],
    7:[1.1645, -4.2015, 3.1190],
    8:[1.4460, -0.5915, 3.6085],
    9:[2.4195, 0.3075, 8.0715],
    10:[-18.6730, 12.1915, -1.4635],
    11:[5.3370, 5.8855, 25.6115],
    12:[4.9290, -2.4800, -13.2920],
    13:[-0.2270, 0.7950, -2.9675],
    14:[-8.4675, -0.6995, -1.6145],
    15:[9.0710, 20.9360, -2.1190],
    16:[1.4265, -2.5305, 17.1890],
    17:[7.0535, -28.1320, 0.0420],
    18:[0.0460, -2.1040, 0.3500],
    19:[10.5180, -1.9640, -0.4745],
    20:[-0.3950, -10.4130, 0.1620],
    21:[-0.0805, 0.0805, -8.2435],
}

def modelShiftBopYcbv(mat, object_id, inverse=False):
    shift = np.array(YCBV_MODEL_SHIFTS[object_id]).reshape((-1,1))
    R = mat[:3, :3]
    t = mat[:3, 3]
    if inverse:
        mat[:3, 3] = t*1000 + R.dot(shift).flatten() # From milimeter to meter
    else:
        mat[:3, 3] = t*1000 - R.dot(shift).flatten() # From milimeter to meter
    return mat

''' Backup '''
# def vectorize(x, use_hsv=True, feature_inliers=True, norm_cos_weight=False, fs_thresh=0.02):
#     p_mask = x['valid_proj']
#     d_valid = x['valid_depth']
#
#     if norm_cos_weight:
#         p_mask = p_mask * x['norm_cos']
#
#     d_mask = (p_mask * d_valid)
#
#     depth_err = x['depth_err'] ##
#     depth_err_distance = (depth_err * d_mask)
#     close_mask = (torch.abs(depth_err) <  fs_thresh) * d_mask
#     close_distance = depth_err * close_mask
#
#     freespace_error = (depth_err - fs_thresh).clamp(min=0) ##
#     freespace_distance = (freespace_error * d_mask)
#     freespace_mask = (freespace_error > 0) * d_mask
#
#     occlusion_error = (depth_err + fs_thresh).clamp(max=0) ##
#     occlusion_distance = (occlusion_error * d_mask)
#     occlusion_mask = (occlusion_error < 0) * d_mask
#
#     color_err = x['color_err'] ##
#     color_err_distance = (color_err*p_mask)
#
#     color_cos = x['color_cos'] ##
#     color_cos_distance = (color_cos*p_mask)
#
#     color_err = x['color_err'] ##
#     color_err_distance_dmask = (color_err*close_mask)
#
#     color_cos = x['color_cos'] ##
#     color_cos_distance_dmask = (color_cos*close_mask)
#
#     p_count = p_mask.float().sum(-1).clamp(min=1e-9)
#     d_count = d_mask.float().sum(-1).clamp(min=1e-9)
#     close_count = close_mask.float().sum(-1).clamp(min=1e-9)
#
#     edge_scores = x['edge_scores']
#     edge_scores_d = x['edge_scores_d']
#
#     score_vec = torch.stack([p_count,
#                              d_count,
#                              close_count,
#                              depth_err_distance.float().sum(-1),
#                              close_distance.float().sum(-1),
#                              freespace_distance.float().sum(-1),
#                              freespace_mask.float().sum(-1),
#                              occlusion_distance.float().sum(-1),
#                              occlusion_mask.float().sum(-1),
#                              color_err_distance.float().sum(-1),
#                              color_cos_distance.float().sum(-1),
#                              color_err_distance_dmask.float().sum(-1),
#                              color_cos_distance_dmask.float().sum(-1),
#                              p_mask.float().mean(-1),
#                              d_mask.float().mean(-1),
#                              close_mask.float().mean(-1),
#                              close_count/d_count,
#                              depth_err_distance.float().sum(-1)/d_count,
#                              close_distance.float().sum(-1)/d_count,
#                              freespace_distance.float().sum(-1)/d_count,
#                              freespace_mask.float().sum(-1)/d_count,
#                              occlusion_distance.float().sum(-1)/d_count,
#                              occlusion_mask.float().sum(-1)/d_count,
#                              color_err_distance.float().sum(-1)/p_count,
#                              color_cos_distance.float().sum(-1)/p_count,
#                              color_err_distance_dmask.float().sum(-1)/close_count,
#                              color_cos_distance_dmask.float().sum(-1)/close_count,
#                              edge_scores.float(),
#                              edge_scores_d.float()
#                             ], axis=-1)
#
#     if use_hsv:
#         h_err = (x['h_err'].float() * close_mask).sum(-1) / close_count
#         s_err = (x['s_err'].float() * close_mask).sum(-1) / close_count
#         v_err = (x['v_err'].float() * close_mask).sum(-1) / close_count
#
#         score_vec = torch.cat([
#             score_vec,
#             h_err.unsqueeze(-1).float(),
#             s_err.unsqueeze(-1).float(),
#             v_err.unsqueeze(-1).float(),
#         ], axis=-1)
#
#     '''For feature inliers'''
#     if feature_inliers:
#         inlier_th = 0.02
#         if ("inlier_feature_dists" not in x) and ("obs_feature_match_count" not in x):
#             assert "corre_dists" in x
#             unique_corre = x['unique_corre']
#             unique_feat_dist = x['unique_feat_dist']
#             corre_dists = x['corre_dists']
#             N = corre_dists.shape[0]
#
#             # Compute the Euclidean and feature distances for the projected feature inliers
#             corre_inlier_mask = corre_dists < inlier_th
#             inlier_feature_dists = []
#             inlier_euclidean_dists = []
#             inlier_idxs = []
#             for i in range(N):
#                 inlier_feature_dists.append(unique_feat_dist[corre_inlier_mask[i]])
#                 inlier_euclidean_dists.append(corre_dists[i, corre_inlier_mask[i]])
#                 inlier_idxs.append(unique_corre[:, corre_inlier_mask[i]])
#
#             x['inlier_feature_dists'] = inlier_feature_dists
#             x['inlier_euclidean_dists'] = inlier_euclidean_dists
#             x['inlier_idxs'] = inlier_idxs
#
#         if 'obs_feature_match_count' not in x:
#             inlier_feature_dists = x['inlier_feature_dists']
#             inlier_euclidean_dists = x['inlier_euclidean_dists']
#             inlier_idxs = x['inlier_idxs']
#
#             inlier_count = torch.tensor([len(_) for _ in inlier_feature_dists], device=color_err.device).float()
#             obs_feature_match_count = torch.tensor([len(np.unique(_[0])) for _ in inlier_idxs], device=color_err.device).float()
#             obs_feature_match_ratio = obs_feature_match_count / obs_feature_match_count.max() # We didn't record the number of obs features
#             inlier_feature_dist_sum = torch.tensor([sum(_) for _ in inlier_feature_dists], device=color_err.device).float()
#             inlier_feature_dist_mean = inlier_feature_dist_sum / inlier_count
#             inlier_euclidean_dist_sum = torch.tensor([sum(_) for _ in inlier_euclidean_dists], device=color_err.device).float()
#             inlier_euclidean_dist_mean = inlier_euclidean_dist_sum / inlier_count
#         else:
#             inlier_count = x['inlier_count'].float()
#             obs_feature_match_count = x['obs_feature_match_count'].float()
#             # obs_feature_match_ratio = x['obs_feature_match_ratio'].float()
#             proj_area = x['proj_area'].float().squeeze()
#             obs_feature_match_ratio = obs_feature_match_count / (proj_area + 1)
#             inlier_feature_dist_sum = x['inlier_feature_dist_sum'].float()
#             inlier_feature_dist_mean = x['inlier_feature_dist_mean'].float()
#             inlier_euclidean_dist_sum = x['inlier_euclidean_dist_sum'].float()
#             inlier_euclidean_dist_mean = x['inlier_euclidean_dist_mean'].float()
#
#         score_vec = torch.cat([
#             score_vec,
#             inlier_count.unsqueeze(-1),
#             obs_feature_match_count.unsqueeze(-1),
#             obs_feature_match_ratio.unsqueeze(-1),
#             inlier_feature_dist_sum.unsqueeze(-1),
#             inlier_feature_dist_mean.unsqueeze(-1),
#             inlier_euclidean_dist_sum.unsqueeze(-1),
#             inlier_euclidean_dist_mean.unsqueeze(-1),
#         ], axis=-1)
#
#     # eliminate NaN in the input vectors
#     score_vec[score_vec != score_vec] = 0
#
#     return score_vec
