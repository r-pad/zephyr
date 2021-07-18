import cv2
import numpy as np
from zephyr.full_pipeline.hog_orientation import computeHOGKeypoints, computeHOGPeaks

class Sift(object):
    def __init__(self, feature_size = 5, step_size = None):
        self.feature_size = feature_size

        if(step_size is None):
            self.step_size = feature_size
        else:
            self.step_size = step_size

        try:
            self.feature = cv2.xfeatures2d.SIFT_create()
        except:
            self.feature = cv2.SIFT()

    def __call__(self, img, depth=None, meta_data=None,
                 mask=None, keypoints = None,
                 detect_keypoints=False,
                 compute_orientations=True):

        if(detect_keypoints):
            keypoints = self.feature.detect(img, mask)

        if(keypoints is None):
            keypoints = computeHOGPeaks(img, self.feature_size,
                step_size=self.step_size, mask=mask)
        else:
            if isinstance(keypoints, np.ndarray):
                keypoints = [cv2.KeyPoint(x=kp[1], y=kp[0], \
                    _size=self.feature_size) for kp in keypoints]
            if(compute_orientations):
                keypoints, _ = computeHOGKeypoints(img, self.feature_size,
                    keypoints)


        keypoints, features = self.feature.compute(img, keypoints)
        features /= np.linalg.norm(features, axis=1).reshape(-1, 1)

        return keypoints, features
