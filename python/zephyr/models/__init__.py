import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

import time
from functools import partial

from .linear import MLP, LogReg
from .pointnet import PointNet
from .pointnet2 import PointNet2SSG
from .pointnet3 import PointNet3SSG
from .dgcnn import DGCNN
# from .masked_conv import ConvolutionalPoseModel
from .point_mlp import PointMLP

from pytorch_lightning.core.lightning import LightningModule

def getModel(model_name, args, mode="train"):
    if args.resume_path is None or mode == 'train':
        if model_name == 'mlp':
            model = MLP(args.dim_agg, args)
        if model_name == "pmlp":
            model = PointMLP(args.dim_point, args)
        elif model_name[:2] == 'lg':
            model = LogReg(args.dim_agg, args)
        elif model_name == "pn":
            model = PointNet(args.dim_point, args)
        elif model_name == "pn2":
            model = PointNet2SSG(args.dim_point, args, num_class=1)
        elif model_name == "pn3":
            model = PointNet3SSG(args.dim_point, args, num_class=1)
        elif model_name == "dgcnn":
            model = DGCNN(args.dim_point, args, num_class=1)
        # elif model_name == "maskconv":
        #     model = ConvolutionalPoseModel(args)
        else:
            raise Exception("Unknown model name:", model_name)
    else:
        if model_name == 'mlp':
            model = MLP.load_from_checkpoint(args.resume_path, args.dim_agg, args)
        elif model_name == "pmlp":
            model = PointMLP.load_from_checkpoint(args.resume_path, args.dim_point, args)
        elif model_name[:2] == 'lg':
            model = LogReg.load_from_checkpoint(args.resume_path, args.dim_agg, args)
        elif model_name == "pn":
            model = PointNet.load_from_checkpoint(args.resume_path, args.dim_point, args)
        elif model_name == "pn2":
            model = PointNet2SSG.load_from_checkpoint(args.resume_path, args.dim_point, args, num_class=1)
        elif model_name == "pn3":
            model = PointNet3SSG.load_from_checkpoint(args.resume_path, args.dim_point, args, num_class=1)
        elif model_name == "dgcnn":
            model = DGCNN.load_from_checkpoint(args.resume_path, args.dim_point, args, num_class=1)
        # elif model_name == "maskconv":
        #     model = ConvolutionalPoseModel.load_from_checkpoint(args.resume_path, args)
        else:
            raise Exception("Unknown model name:", model_name)
    if not args.pretrained_pnfeat is None:
        model.loadPretrainedFeat(args.pretrained_pnfeat)
    return model
