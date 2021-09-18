import numpy as np

from zephyr.full_pipeline.feature_frames import featurize
from zephyr.full_pipeline.sift import Sift

def featurizeScene(img, depth, mask, meta_data, feature_sizes, feature_steps, feature_type='sift', smart_scale=False):
    keypoints, features, cloud, frames = [], [], [], []
    for feature_size, feature_step in zip(feature_sizes, feature_steps):
        if smart_scale:
            avg_depth = depth.mean() / 10000.0
            feature_size = int(feature_size * (1/avg_depth))
            feature_step = int(feature_step * (1/avg_depth))
        if feature_type == "sift":
            featurizer = Sift(feature_step, feature_size)
            keypoints_scale, features_scale, cloud_scale, frames_scale = \
                featurize(img, depth, mask, meta_data, featurizer)
        elif feature_type == "lfnet":
            keypoints_scale, features_scale, cloud_scale, frames_scale = \
                featurize(img, depth, mask, meta_data, lfnet)
        elif feature_type == "lfnet-sift":
            featurizer = MixFeature(feature_size=feature_size, feature_step=feature_step,
                ori_type="lfnet", feature_type="sift", lfnet=lfnet)
            keypoints_scale, features_scale, cloud_scale, frames_scale = \
                featurize(img, depth, mask, meta_data, featurizer)
        elif feature_type == "sift-lfnet":
            featurizer = MixFeature(feature_size=feature_size, feature_step=feature_step,
                ori_type="sift", feature_type="lfnet", lfnet=lfnet)
            keypoints_scale, features_scale, cloud_scale, frames_scale = \
                featurize(img, depth, mask, meta_data, featurizer)
        keypoints += keypoints_scale
        features.append(features_scale)
        cloud.append(cloud_scale)
        frames.append(frames_scale)

    features = np.concatenate(features, axis=0)
    cloud = np.concatenate(cloud, axis=0)
    frames = np.concatenate(frames, axis=0)
    
    return keypoints, features, cloud, frames
