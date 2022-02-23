import os
import numpy as np
import open3d as o3d
import torch

from zephyr.full_pipeline.model_grid import loadGridData, bopProjectModelPC, bopLoadGridKeypoints, getModelPointsCloud
from zephyr.full_pipeline.sample_mesh import sampleMeshPointCloud
from zephyr.full_pipeline.feature_match import samplePoseHypos
from zephyr.full_pipeline.feature_frames import compute_transforms
from zephyr.full_pipeline.corre_score import knnDistScore
from zephyr.full_pipeline.feature_pyramid import MultiscaleGrid
from zephyr.full_pipeline.sift import Sift

from zephyr.utils import to_np, getClosestTrans


from bop_toolkit_lib.misc import ensure_dir

from object_pose_utils.utils.pose_error import add, adi


class FeatureModel():
    def __init__(self, dataset_root, is_sym, args, create_index=False):
        self.dataset_root = dataset_root
        self.grid_dir_name = args.grid_dir_name
        self.sampled_model_dir_name = args.sampled_model_dir_name
        self.real_model_dir_name = args.real_model_dir_name
        self.leaf_size = args.sampled_model_leaf_size
        self.n_samples = args.sampled_model_n_samples
        self.sift_feature_size = args.model_sift_feature_size
        self.feature_index_gpu = args.feature_index_gpu

        self.num_sampled_trans = args.num_sampled_trans
        self.K = args.feature_match_K
        self.uniform_sampling = args.uniform_sampling
        self.oracle_sampling = args.oracle_sampling

        self.grid_indices = np.load(args.grid_indices_path)
        self.grid_projection_dir_name = '%s_projection' % self.grid_dir_name
        self.grid_data_dir_name = "%s_data" % self.grid_dir_name

        self.is_sym = is_sym
        self.err_func = adi if self.is_sym else add
        self.create_index = create_index

    def construct(self, obj_id, model_path, dataset_camera):
        dataset_root = self.dataset_root
        grid_dir_name = self.grid_dir_name
        sampled_model_dir_name = self.sampled_model_dir_name
        leaf_size = self.leaf_size
        n_samples = self.n_samples
        grid_indices = self.grid_indices
        grid_projection_dir_name = self.grid_projection_dir_name
        grid_data_dir_name = self.grid_data_dir_name
        feature_grid_scales = [self.sift_feature_size]

        '''Down-sample the object mesh model'''
        sampled_model_pc_path = os.path.join(dataset_root, sampled_model_dir_name, "%s.pcd" % model_path.split('/')[-1].split('.')[0])
        if not os.path.exists(sampled_model_pc_path):
            print("sampling...")
            ensure_dir(os.path.dirname(sampled_model_pc_path))
            sampleMeshPointCloud(model_path, sampled_model_pc_path, n_samples, leaf_size, write_normals=True)
        
        '''load the sampled model'''
        sampled_model_pcd = o3d.io.read_point_cloud(sampled_model_pc_path)
        sampled_model_pc = np.asarray(sampled_model_pcd.points) / 1000.0 # Convert the model point cloud into meters
        sampled_model_normals = np.asarray(sampled_model_pcd.normals)

        sampled_model_pc_normals = np.concatenate((sampled_model_pc, sampled_model_normals), axis=1)

        '''Get the RGBD renders in grid'''
        grid_folder = os.path.join(dataset_root, grid_dir_name)
        grid_images, grid_depths, grid_masks, grid_quats, grid_metas = loadGridData(grid_folder, grid_indices, obj_id, return_meta=True)

        '''Get the projected model points for each render'''
        grid_pc_folder = os.path.join(dataset_root, grid_projection_dir_name, "%06d" % obj_id)
        bopProjectModelPC(sampled_model_pc_normals, grid_indices, grid_depths, grid_metas, dataset_camera, grid_pc_folder)

        '''Load the projected model points'''
        keypoints_all, pc_normals_all, kpt_idx_all = bopLoadGridKeypoints(grid_pc_folder, grid_indices, leaf_size)
        grid_kpts = [_[:, :3] for _ in pc_normals_all]
        grid_normals = [_[:, 3:6] for _ in pc_normals_all]

        '''SIFT featurization on object model'''
        grid_data_dir = os.path.join(dataset_root, grid_data_dir_name)
        ensure_dir(grid_data_dir)
        grid = MultiscaleGrid(
            grid_images, grid_depths, grid_masks, grid_quats,
            keypoints_all, grid_kpts, grid_normals, kpt_idx_all, scales=feature_grid_scales,
            featurize_class=Sift, id=obj_id, path_root=grid_data_dir
        )
        grid_features = np.concatenate([grid.features[s] for s in feature_grid_scales], axis=0).astype('float32').copy(order='C')
        grid_clouds = np.concatenate([grid.clouds[s] for s in feature_grid_scales], axis=0)
        grid_frames = np.concatenate([grid.frames[s] for s in feature_grid_scales], axis=0)
        grid_kpt_idx = np.concatenate([grid.indices[s] for s in feature_grid_scales], axis=0)
        grid_harris_scores = np.concatenate([grid.harris_scores[s] for s in feature_grid_scales], axis=0)

        if self.create_index:
            feature_index = createFeatureIndex(grid_features, gpu = self.feature_index_gpu)
            self.feature_index = feature_index

        '''Get model norms and colors by combining mesh renders and point renders'''
        model_points, model_normals, model_colors = getModelPointsCloud(
            sampled_model_pc, sampled_model_normals, grid_images, keypoints_all, kpt_idx_all
        )

        if self.real_model_dir_name is not None:
            real_model_pc_path = os.path.join(dataset_root, self.real_model_dir_name, "%s.pcd" % model_path.split('/')[-1].split('.')[0])
            real_model_pcd = o3d.io.read_point_cloud(real_model_pc_path)
            model_points = np.asarray(sampled_model_pcd.points) / 1000.0


        self.model_points = model_points
        self.model_normals = model_normals
        self.model_colors = model_colors
        self.grid_frames = grid_frames
        self.grid_clouds = grid_clouds
        self.grid_kpt_idx = grid_kpt_idx
        self.grid_quats = grid_quats

    def getModelData(self):
        return self.model_points, self.model_normals, self.model_colors

    def match(self, features, frames, mat_gt):
        if not self.create_index:
            raise Exception("Index not created")
        feature_index = self.feature_index
        grid_frames = self.grid_frames
        grid_clouds = self.grid_clouds
        K = self.K
        num_sampled_trans = self.num_sampled_trans
        uniform_sampling = self.uniform_sampling
        grid_kpt_idx = self.grid_kpt_idx

        '''Match obs kps to model kps'''
        knn_dists, knn_idxs = feature_index.search(features, k=K)
        target_idxs = np.repeat(np.arange(knn_idxs.shape[0]),
                                knn_idxs.shape[1])
        source_idxs = knn_idxs.flatten()
        target_frames = np.array(frames)
        source_frames = np.array(grid_frames)
        transforms = compute_transforms(torch.from_numpy(source_frames[source_idxs]),
                                        torch.from_numpy(target_frames[target_idxs]))
        transforms = to_np(transforms)
        corre_points = grid_clouds[knn_idxs,:]

        '''Sampling the hypotheses'''
        corre_scores = knnDistScore(knn_dists)

        sample_idxs = samplePoseHypos(corre_scores, num_sampled_trans, uniform_sampling)
        sample_trans = transforms[sample_idxs]
        sample_scores = corre_scores[sample_idxs]

        '''Get the oracle hypotheses and combine them'''
        if self.oracle_sampling:
            min_idx = getClosestTrans(transforms, mat_gt)
            oracle_trans = transforms[min_idx]

            sample_trans = np.concatenate([sample_trans, np.expand_dims(oracle_trans, 0)], axis=0)
            
        obs_idxs = np.repeat(np.arange(knn_idxs.shape[0]),
                             knn_idxs.shape[1]).reshape(knn_idxs.shape)
        grid_idxs = grid_kpt_idx[knn_idxs]

        match_aux = {
            "grid_idxs": grid_idxs,
            "obs_idxs": obs_idxs,
            "knn_dists": knn_dists,
            "target_frames": target_frames
        }

        return sample_trans, match_aux


def createFeatureIndex(grid_features, gpu=None):
    import faiss
    # Set up Grid Search Index
    N, D = grid_features.shape

    if(gpu in range(4)):
        gpu_res = faiss.StandardGpuResources()
        feature_index_cpu = faiss.IndexFlatL2(D)
        feature_index = faiss.index_cpu_to_gpu(gpu_res, gpu, feature_index_cpu)
    else:
        feature_index = faiss.IndexFlatL2(D)

    feature_index.add(grid_features)

    return feature_index
