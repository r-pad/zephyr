import numpy as np

def samplePoseHypos(corre_scores, num_samples=1000, uniform_sampling=False):
    # Select the transforms that have highest corr score
    if(num_samples is not None and \
       num_samples < len(corre_scores)):
        if(uniform_sampling):
            p = None
        else:
            p=corre_scores.copy()
            p /= p.sum()
        sample_idxs = np.random.choice(len(corre_scores), num_samples,
                                       replace=False, p=p)
    else:
        sample_idxs = np.arange(len(corre_scores))

    return sample_idxs
