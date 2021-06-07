'''
This python file implements a wrapper for surface matching (PPF) algorithm in Halcon software
'''
import time
import numpy as np
from scipy.spatial.transform import Rotation as R

'''
Please refer https://pypi.org/project/mvtec-halcon/ for setting up Halcon/Python Interface. 
More documentations can be found here: https://www.mvtec.com/products/halcon/documentation
Note that the Halcon software of the corresponding version must be installed
'''
import halcon as ha

class PPFModel():
    def __init__(
            self, object_model,
            ModelSamplingDist = 0.03,
            ModelInvertNormals = 'true',
            UseViewBased = 'false',
        ) -> None:
        '''
        @params
        object_model: str or np.ndarray. The path to the ply model file or the array storing model points
                      XYZ coordinates of the model points should be in milimeter
        others: all other arguments are to be used in create_surface_model() in halcon
                Refer https://www.mvtec.com/doc/halcon/2105/en/create_surface_model.html for details
        '''
        # Create the Object3DModel
        if type(object_model) is str:
            Object3DModel, StatusModel = ha.read_object_model_3d(object_model, 'm', [], [])
        elif type(object_model) is np.ndarray:
            Object3DModel = ha.gen_object_model_3d_from_points(object_model[:, 0].tolist(), object_model[:, 1].tolist(), object_model[:, 2].tolist())
            Object3DModel = ha.surface_normals_object_model_3d(Object3DModel, 'mls', [], [])
        else:
            raise Exception("Unknown type of object_model:", type(object_model))
        
        self.Object3DModel = Object3DModel
        
        self.ObjectSurfaceModel = ha.create_surface_model(
            self.Object3DModel, ModelSamplingDist, 
            ['model_invert_normals', 'train_view_based'],
            [ModelInvertNormals, UseViewBased], 
        )
        
        self.UseViewBased = UseViewBased

    def find_surface_model(
            self, scene_pc, 
            MaxOverlapDistRel=0, 
            NumResult=100,
            SceneNormalComputation='mls',
            SparsePoseRefinement = 'true',
            DensePoseRefinement = 'true',
            RefPtRate = 1,
            SceneSamplingDist = 0.03,
        ):
        '''
        @params
        scene_pc: np.ndarray of shape (N, 6), the scene point cloud, in milimeter
        others: all other arguments are to be used in find_surface_model() in halcon
                Refer https://www.mvtec.com/doc/halcon/2105/en/find_surface_model.html for details

        @return
        poses_ppf: np.ndarray of shape (NumResult, 4, 4), the estimated poses, in milimeter
        scores_ppf: list of length NumResult, the score of each pose given by PPF algorithm
        time_ppf: float, the time used by find_surface_model()
        '''
        Scene3DModel = ha.gen_object_model_3d_from_points(scene_pc[:, 0].tolist(), scene_pc[:, 1].tolist(), scene_pc[:, 2].tolist())
        
        t1 = time.time()
        Pose, Score, SurfaceMatchingResultID = ha.find_surface_model(
            self.ObjectSurfaceModel, Scene3DModel, SceneSamplingDist, RefPtRate, 0, 'true', 
            ['num_matches', 'use_view_based', 'max_overlap_dist_rel', 'scene_normal_computation', 'sparse_pose_refinement', 'dense_pose_refinement'], \
            [NumResult, self.UseViewBased, MaxOverlapDistRel, SceneNormalComputation, SparsePoseRefinement, DensePoseRefinement],
        )
        t2 = time.time()
        
        poses_raw = np.asarray(Pose).reshape((NumResult, 7))

        poses_rot = R.from_euler("XYZ", poses_raw[:, 3:6], degrees=True)
        poses_rotmat = poses_rot.as_matrix()
        poses_ppf = np.zeros((NumResult, 4, 4))
        poses_ppf[:, :3, :3] = poses_rotmat
        poses_ppf[:, :3, 3] = poses_raw[:, :3]
        poses_ppf[:, 3, 3] = 1

        scores_ppf = Score
        time_ppf = t2 - t1

        return poses_ppf, scores_ppf, time_ppf