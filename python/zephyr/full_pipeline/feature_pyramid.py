import numpy as np
import cv2
import torch
import os, pickle

from quat_math import quaternion_matrix
from zephyr.full_pipeline.feature_frames import compute_frames
from zephyr.full_pipeline.keypoints import keypoints2Indices
from zephyr.full_pipeline.sift import Sift
from zephyr.full_pipeline.corre_score import HarrisScore

def torch2Img(img, normalized = False):
    disp_img = to_np(img)
    if len(disp_img.shape) == 4:
        disp_img = disp_img[0]
    disp_img = disp_img.transpose((1,2,0))
    if(normalized):
        mean = np.array([0.485, 0.456, 0.406])
        std = np.array([0.229, 0.224, 0.225])
        disp_img = disp_img * std + mean
    return disp_img

class MultiscaleGrid(object):
    def __init__(self, images, depths, masks, quats,
                 keypoints, points, normals, point_ids,
                 scales, featurize_class = Sift, featurizer_instance=None,
                 id = None, path_root = "/datasets/ycb/grid_data",
                 ):

        fx = 1066.778
        fy = 1067.487
        px = 312.9869
        py = 241.3109

        self.meta_data = {}
        self.meta_data['camera_scale'] = 10000
        self.meta_data['camera_fx'] = fx
        self.meta_data['camera_fy'] = fy
        self.meta_data['camera_cx'] = px
        self.meta_data['camera_cy'] = py

        self.id = id
        self.path_root = path_root
        self.scales = scales

        if not id is None:
            self.file_path = os.path.join(self.path_root, "%03d_%s.pk" % (self.id, "-".join([str(_) for _ in self.scales])))

        if (not id is None) and os.path.exists(self.file_path):
            ''' Load the pre-generated feature pyramid '''
            self.loadData(self.file_path)
        else:
            self.indices = {s:[] for s in scales}
            self.features = {s:[] for s in scales}
            self.frames = {s:[] for s in scales}
            self.clouds = {s:[] for s in scales}
            self.view_ids = {s:[] for s in scales}

            self.keypoint_features = {s:{} for s in scales}
            self.keypoint_quats = {s:{} for s in scales}

            # feature descriptors that are viewed from the most straight views
            self.most_straight_features = {s:None for s in scales}
            self.most_straight_cos = {s: None for s in scales}
            self.view_kpt_features = {s: [{} for _ in range(len(keypoints))] for s in scales}
            self.n_kpts = max([_.max() for _ in point_ids]) + 1 # keypoint id starts from 0

            self.harris_scores = {s:[] for s in scales}

            for s in scales:
                if featurizer_instance is None:
                    featurizer = featurize_class(feature_size = s)
                else:
                    featurizer = featurizer_instance
                    featurizer.feature_size = s
                    featurizer.step_size = s

                for i_view, (kps, ids, pts, norms, img, depth, mask, q) in \
                    enumerate(zip(keypoints, point_ids, points, normals,
                        images, depths, masks, quats)):

                    trans_mat = quaternion_matrix(q)
                    trans_mat[:3,3] = np.array([0,0,1])
                    trans_inv = np.linalg.inv(trans_mat)

                    view_normals = norms.dot(trans_mat[:3,:3])
                    view_points = (trans_mat[:3,:3].dot(pts.T) + trans_mat[:3,3:]).T

                    # corresponding to features in view_points
                    cam_norm_cos = (view_normals * -view_points).sum(-1) / \
                                   (np.linalg.norm(view_normals, axis=-1) * np.linalg.norm(view_points, axis=-1))

                    valid_norms = view_normals[:,2] < 0
                    if valid_norms.sum() <= 0:
                        print("Error! Still some valid_norms.sum()<=0")
                        continue
                    view_keypoints, view_features = featurizer(img, depth, self.meta_data, mask,
                        keypoints=kps[valid_norms])

                    dim_feat = view_features.shape[1]

                    # Recompute the map from view_keypoints/view_features to view_points/view_normals as render_kpids
                    # Handle that some keypoints are added or removed during the featurization process
                    idxs_map = keypoints2Indices(kps, return_map=True, size=img.shape)
                    kp_idxs = keypoints2Indices(np.fliplr(cv2.KeyPoint_convert(view_keypoints)),
                        size=img.shape)
                    render_kpids = []
                    for i, k in enumerate(kp_idxs):
                        success = False
                        for shifted_k in [k, k+1, k-1, k+img.shape[0], k-img.shape[0]]:
                            if shifted_k in idxs_map:
                                render_kpids.append(idxs_map[shifted_k])
                                success = True
                                break
                        if success:
                            continue
                        else:
                            raise Exception("Cannot find close KeyPoint. ")

                    '''Calculate the Harris scores for each keypoints'''
                    self.harris_scores[s].append(HarrisScore(view_keypoints, img))

                    orientations = [kp.angle*np.pi/180. for kp in view_keypoints]

                    # Record the feature if it is the by far the most straight view of the keypoint
                    if self.most_straight_features[s] is None:
                        self.most_straight_features[s] = np.zeros((self.n_kpts, dim_feat))
                        self.most_straight_cos[s] = np.ones(self.n_kpts) * -1
                    else:
                        for i_view_keypoints, i_view_points in enumerate(render_kpids):
                            true_kpt_id = ids[i_view_points]
                            self.view_kpt_features[s][i_view][true_kpt_id] = view_features[i_view_keypoints]
                            if cam_norm_cos[i_view_points] > self.most_straight_cos[s][true_kpt_id]:
                                self.most_straight_cos[s][true_kpt_id] = cam_norm_cos[i_view_points]
                                self.most_straight_features[s][true_kpt_id] = view_features[i_view_keypoints]

                    frames = compute_frames(view_points[render_kpids], view_normals[render_kpids], orientations)
                    for k, frm in enumerate(frames):
                        frames[k] = trans_inv.dot(frm)

                    self.indices[s].append(ids[render_kpids])
                    self.features[s].append(view_features)
                    self.frames[s].append(frames)
                    self.clouds[s].append(pts[render_kpids])
                    self.view_ids[s].append(np.ones(len(view_features)) * i_view)

                    for j, feat in zip(ids[render_kpids], view_features):
                        if(j not in self.keypoint_features[s].keys()):
                            self.keypoint_features[s][j] = []
                            self.keypoint_quats[s][j] = []
                        self.keypoint_features[s][j].append(feat)
                        self.keypoint_quats[s][j].append(q)

                for j in self.keypoint_features[s].keys():
                    self.keypoint_features[s][j] = np.stack(self.keypoint_features[s][j])
                    self.keypoint_quats[s][j] = np.stack(self.keypoint_quats[s][j])

                self.features[s] = np.concatenate(self.features[s])
                self.indices[s] = np.concatenate(self.indices[s])
                self.clouds[s] = np.concatenate(self.clouds[s])
                self.frames[s] = np.concatenate(self.frames[s])
                self.harris_scores[s] = np.concatenate(self.harris_scores[s])
                self.view_ids[s] = np.concatenate(self.view_ids[s])

            if not id is None:
                '''Save the featurized grid data'''
                self.saveData(self.file_path)

    def saveData(self, file_path):
        print("Saving data to", file_path)
        data = {
            "indices": self.indices,
            "features": self.features,
            "clouds": self.clouds,
            "frames": self.frames,
            "harris_scores": self.harris_scores,
            "keypoint_features": self.keypoint_features,
            "keypoint_quats": self.keypoint_quats,
            "most_straight_features": self.most_straight_features,
            "most_straight_cos": self.most_straight_cos,
            "view_kpt_features": self.view_kpt_features,
            "view_ids": self.view_ids,
        }
        pickle.dump(data, open(file_path, "wb"))

    def loadData(self, file_path):
        print("Loading data from", file_path)
        data = pickle.load(open(file_path, "rb"))
        self.indices = data["indices"]
        self.features = data["features"]
        self.clouds = data["clouds"]
        self.frames = data["frames"]
        self.harris_scores = data["harris_scores"]
        self.keypoint_features = data["keypoint_features"]
        self.keypoint_quats = data["keypoint_quats"]
        if 'most_straight_features' in data:
            self.most_straight_features = data['most_straight_features']
            self.most_straight_cos = data['most_straight_cos']
            self.view_kpt_features = data['view_kpt_features']
            self.view_ids = data['view_ids']
