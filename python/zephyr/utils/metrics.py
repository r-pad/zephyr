import torch
import numpy as np
from zephyr.constants import OBJECT_DIAMETERES


class ErrAucMetric():
    def __init__(self, dataset_name, max_err=0.5, step=0.05):
        self.obj_diam = OBJECT_DIAMETERES[dataset_name]

    def __call__(self, errs, obj_ids):

        pass

class AddErrorMetric():
    def __init__(self, dataset_name, threshold = 0.1):
        self.obj_diam = OBJECT_DIAMETERES[dataset_name]
        self.threshold = threshold

    def __call__(self, errs, obj_ids):
        errs = list(errs)
        obj_ids = list(obj_ids)
        assert len(errs) == len(obj_ids)
        err_thresholds = self.threshold * torch.tensor([self.obj_diam[i] for i in obj_ids])
        return (torch.tensor(errs) <= err_thresholds).float().sum() / len(errs)