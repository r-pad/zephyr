import os
import numpy as np
import torch
import torch.cuda
import random

from pytorch_lightning import Trainer

from zephyr.models import getModel
from zephyr.datasets import getDataloader
from zephyr.options import getOptions, checkArgs

def main(args):
    assert len(args.dataset_root) == 1
    np.random.seed(0)
    random.seed(0)
    torch.manual_seed(0)

    args.exp_name = "_".join([args.model_name, args.dataset, args.exp_name])
    args.test_dataset = True
    print("exp_name:", args.exp_name)
    # assert args.exp_name in args.resume_path

    if args.dataset_name[0] == "ycbv":
        print("args.icp = True")
        args.icp = True
        # args.dataset_root[0] = "/home/qiaog/datasets/ycb/matches_data_test/"
    if args.dataset_name[0] == "lmo":
        print("args.inconst_ratio_th = 100")
        args.inconst_ratio_th = 100
        args.ppf_only = True
        # args.dataset_root[0] = "/home/qiaog/datasets/bop/lmo/grid_0.7m_bop_test_drost100_match_data/"
        # args.n_ppf_hypos = 100
    if args.model_name == "pn":
        args.chunk_size = 1101

    '''Dataloader'''
    boptest_loader = getDataloader(args)

    '''Model'''
    if args.resume_path is None:
        raise Exception("The path to pretrained model must be provided in test.py")

    print("############ BOP test set: %d ##############" % len(boptest_loader))
    args.data_split = args.dataset_name[0] + "-boptest"
    model = getModel(args.model_name, args, mode='test')
    trainer = Trainer(
        gpus=args.n_gpus, distributed_backend='dp',
        checkpoint_callback=False
    )
    trainer.test(model, boptest_loader)

if __name__ == '__main__':
    parser = getOptions()
    args = parser.parse_args()
    checkArgs(args)
    main(args)
