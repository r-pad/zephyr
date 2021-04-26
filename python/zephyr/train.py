import open3d as o3d
import os
import numpy as np
import torch
import torch.cuda
import random

from pytorch_lightning import Trainer
from pytorch_lightning.loggers import TensorBoardLogger as Logger
from pytorch_lightning.callbacks import ModelCheckpoint

from .models import getModel
from .datasets import getDataloader
from .options import getOptions, checkArgs

def main(args):
    np.random.seed(0)
    random.seed(0)
    torch.manual_seed(0)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    args.exp_name = "_".join([args.model_name, args.dataset, args.exp_name, "_".join(args.dataset_name)])
    # args.exp_name = "_".join([args.model_name, args.dataset, args.exp_name])
    print("exp_name:", args.exp_name)

    if args.model_name == "mlp" or args.model_name[:2] == "lg":
        args.num_workers = 0

    '''Dataloader'''
    train_loader, val_loader = getDataloader(args)

    '''Model'''
    model = getModel(args.model_name, args, mode="train")

    '''Logger and checkpoint'''
    logger = Logger("tb_logs/", name=args.exp_name)
    logger.log_hyperparams(args)
    if args.train_single:
        checkpoint_callback = None
    else:
        checkpoint_callback = ModelCheckpoint(
            filepath = "lightning_logs/%s/{epoch}-{train_epoch_err:.3f}-{val_epoch_err:.3f}" % args.exp_name,
            monitor = "val_epoch_err", save_top_k = -1, mode = "min"
        )

    trainer = Trainer(
        gpus=args.n_gpus, distributed_backend='dp',
        logger=logger, checkpoint_callback=checkpoint_callback,
        resume_from_checkpoint=args.resume_path,
        max_epochs=args.n_epoch
    )

    trainer.fit(model, train_loader, val_loader)

if __name__ == '__main__':
    parser = getOptions()
    args = parser.parse_args()
    checkArgs(args)
    main(args)
