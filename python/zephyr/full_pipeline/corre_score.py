import numpy as np
import cv2
import pickle

def loadModel(path="../scripts/results/sampling/LR_model.pk"):
    model = pickle.load(open(path, "rb"))
    return model



'''Harris Corner score at the given keypoints in the input image'''
def HarrisScore(keypoints, img):
    # print(type(keypoints[0]))
    # print(len(keypoints))
    # print(keypoints[0])
    if type(keypoints[0]) is cv2.KeyPoint:
        keypoints = np.fliplr(cv2.KeyPoint_convert(keypoints))
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray = np.float32(gray)/255.0
    dst = cv2.cornerHarris(gray,5,3,0.04)
    scores = dst[
        keypoints[:, 0].round().astype(int), keypoints[:, 1].round().astype(int)
    ]
    return scores

'''Normalized inverse of the kNN distances'''
def knnDistScore(knn_dists):
    knn_norm = (1/knn_dists) / np.sum(1/knn_dists,
                                      axis=1).reshape(-1, 1)
    corre_scores = knn_norm.flatten()
    return corre_scores

'''How many times an obs point is matched to the same model point in different renders'''
def matchUniqueScore(kpt_idx, k):
    kpt_idx = kpt_idx.reshape((-1, k))
    scores = []
    for i in range(len(kpt_idx)):
        uniques, unique_inverse, unique_counts = np.unique(kpt_idx[i], return_inverse=True, return_counts=True)
        this_scores = unique_counts[unique_inverse]
        scores.append(this_scores)
    scores = np.concatenate(scores).astype(float)
    return scores

'''Difference from the closest distance at which the same obs point is match to any other model point'''
def distDiffScore(knn_dists):
    closest = np.min(knn_dists, axis=1).reshape((-1, 1))
    score = (knn_dists - closest).flatten()
    return score

'''Ratio over the closest distance at which the same obs point is match to any other model point'''
def distRatioScore(knn_dists):
    closest = np.min(knn_dists, axis=1).reshape((-1, 1))
    score = (knn_dists / closest).flatten()
    return score

'''Number of matches within eps of the distance of the top match'''
def thresholdInlierScore(knn_dists, eps=1.5):
    closest = np.min(knn_dists, axis=1).reshape((-1, 1))
    inlier = knn_dists < (eps * closest)
    n_inlier = inlier.sum(axis=1).reshape((-1, 1)) * np.ones_like(knn_dists)
    return n_inlier.flatten(), inlier.flatten()
