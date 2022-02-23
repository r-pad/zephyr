import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from pytorch_lightning.core.lightning import LightningModule

from abc import abstractmethod

from zephyr.losses import getLoss
from zephyr.data_util import convertResultsBOP, saveOutputs
from zephyr.utils import meta2K
from zephyr.utils.icp import icpRefinement
from zephyr.utils.metrics import AddErrorMetric

import csv, os, copy

class BaseModel(LightningModule):
    def __init__(self, args):
        super(BaseModel, self).__init__()
        self.criterion = getLoss(args)

        self.unseen_oids = args.unseen_oids
        self.lr = args.lr
        self.chunk_size = args.chunk_size
        self.n_datasets = len(args.dataset_root)
        self.dataset_name = args.dataset_name

        self.args = args

        self.add01d_metrics = {}
        for dn in self.dataset_name:
            assert type(dn) is str
            self.add01d_metrics[dn] = AddErrorMetric(dn)

    @abstractmethod
    def forward(self, data):
        raise NotImplementedError

    def getRegLoss(self):
        return 0

    def splitChunks(self, data, chunk_size):
        chunks = []
        pp_err = data['pp_err']
        idx_chunks = torch.split(torch.arange(pp_err.shape[0]), chunk_size)
        for idx_chunk in idx_chunks:
            chunk = {}
            for k, v in data.items():
                if type(v) is torch.Tensor and v.nelement() > 1 and \
                        k in ['pp_err', 'transforms'] or k[-2:] == "_x":
                    chunk[k] = v[idx_chunk]
                else:
                    chunk[k] = v
            chunks.append(chunk)
        return chunks

    def train_dataloader(self):
        return None

    def val_dataloader(self):
        return None

    def training_step(self, batch, batch_idx):
        data = batch
        pp_err = data['pp_err']

        if self.chunk_size is not None and pp_err.shape[0] <= self.chunk_size:
            pred_y = self.forward(data)
        else:
            chunks = self.splitChunks(data, self.chunk_size)
            pred_y = []
            for chunk in chunks:
                pred_y_chunk = self.forward(chunk)
                pred_y.append(pred_y_chunk)
            pred_y = torch.cat(pred_y)

        return {"pred_y": pred_y, "pp_err": pp_err}

    def training_step_end(self, outputs):
        pred_y = torch.cat([outputs['pred_y']])
        pp_err = torch.cat([outputs['pp_err']])

        loss = self.criterion(pred_y, pp_err)
        loss += self.getRegLoss()

        gt_idx = torch.argmin(pp_err)
        max_idx = torch.argmax(pred_y)
        gt_err = pp_err[gt_idx]
        max_err = pp_err[max_idx]

        log = {"train_err": max_err, "train_loss": loss}
        return {"loss": loss, 'log': log}

    def training_epoch_end(self, outputs):
        train_epoch_err = 0
        train_epoch_loss = 0
        for o in outputs:
            train_epoch_err += o['train_err']
            train_epoch_loss += o['train_loss']
        train_epoch_err /= len(outputs)
        train_epoch_loss /= len(outputs)

        log = {
            "train_epoch_err": train_epoch_err,
            'train_epoch_loss': train_epoch_loss
        }

        # print("training_epoch_end: train_epoch_err =", train_epoch_err)

        return {
            "train_epoch_err": train_epoch_err,
            "log": log
        }

    def validation_step(self, batch, batch_idx):
        data = batch
        # print("validation_step")
        # for k, v in data.items():
        #     print(k)
        #     print(type(v))
        #     print(v.device)
        # print(data['pp_err'].shape)
        pp_err = data['pp_err']

        # print(transforms.shape)
        # print(pp_err.shape)

        if self.chunk_size is not None and pp_err.shape[0] <= self.chunk_size:
            pred_y = self.forward(data)
        else:
            chunks = self.splitChunks(data, self.chunk_size)
            pred_y = []
            for chunk in chunks:
                pred_y_chunk = self.forward(chunk)
                pred_y.append(pred_y_chunk)
            pred_y = torch.cat(pred_y)

        # if self.args.mask_th is not None:
        #     hypo_mask = data['seg_scores'] > self.args.mask_th
        # else:
        #     hypo_mask = torch.ones_like(pred_y)
        if self.args.use_mask_test:
            hypo_mask = data['mask_scores'] > 0.2
        else:
            hypo_mask = torch.ones_like(pred_y)

        return {
            "pred_y": pred_y, "pp_err": pp_err,
            "transforms": data['transforms'],
            "object_id": data['object_id'][0],
            "scene_id": data['scene_id'][0],
            "im_id": data['im_id'][0],
            "dataset_i": data['dataset_i'][0],
            "hypo_mask": hypo_mask,
            "depth": data['depth'],
            "meta_data": data['meta_data'],
            "uv_original": data['uv_original'],
            "model_points": data['model_points']
        }

    def validation_step_end(self, outputs):
        pred_y = torch.cat([outputs['pred_y']])
        pp_err = torch.cat([outputs['pp_err']])
        transforms = torch.cat([outputs['transforms']])
        hypo_mask = torch.cat([outputs['hypo_mask']])
        uv_original = torch.cat([outputs['uv_original']])

        # assert pp_err[-1] == 0
        assert pp_err[-2] > 0

        loss = self.criterion(pred_y, pp_err)
        loss += self.getRegLoss()

        # Purly based on order
        if self.args.no_model:
            pred_y = - torch.arange(len(pred_y), dtype=pred_y.dtype, device=pred_y.device)

        max_idx = torch.argmax(pred_y)
        max_err = pp_err[max_idx]
        mat = transforms[max_idx]

        if len(pred_y) == 1:
            mat_wo_oracle = torch.eye(4, device = transforms.device, dtype=transforms.dtype)
            max_err_wo_oracle = 1
        else:
            max_idx_wo_oracle = torch.argmax(pred_y[:-1])
            max_err_wo_oracle = pp_err[max_idx_wo_oracle]
            mat_wo_oracle = transforms[max_idx_wo_oracle]

            if self.args.icp:
                cam_K = meta2K(outputs['meta_data'])
                depth = to_np(outputs['depth'])
                model_points = to_np(outputs['model_points'])
                # mat, _ = icpRefinement(depth, to_np(uv_original[max_idx]), to_np(mat), cam_K, model_points, inpaint_depth=False, icp_max_dist=0.01)
                mat_wo_oracle, _ = icpRefinement(depth, to_np(uv_original[max_idx_wo_oracle]), to_np(mat_wo_oracle), cam_K, model_points, inpaint_depth=False, icp_max_dist=0.01)

        if len(outputs['object_id'].shape) >= 1:
            object_id = outputs['object_id'][0]
            scene_id = outputs['scene_id'][0]
            im_id = outputs['im_id'][0]
            dataset_i = outputs['dataset_i'][0]
        else:
            object_id = outputs['object_id'].item()
            scene_id = outputs['scene_id'].item()
            im_id = outputs['im_id'].item()
            dataset_i = outputs['dataset_i'].item()

        return {
            "val_loss": loss,
            "val_err": max_err, "val_err_wo_oracle": max_err_wo_oracle,
            "mat": mat, "mat_wo_oracle": mat_wo_oracle,
            "object_id": object_id,
            "scene_id": scene_id,
            "im_id": im_id,
            "dataset_i": dataset_i,
        }

    def validation_epoch_end(self, outputs_all):
        '''For each dataset'''
        log = {}
        seen_outputs_all = []
        unseen_outputs_all = []
        for dataset_i in range(self.n_datasets):
            unseen_oids = self.unseen_oids[dataset_i]
            dataset_name = self.dataset_name[dataset_i]
            outputs = [o for o in outputs_all if o['dataset_i'] == dataset_i]
            seen_outputs = [o for o in outputs if o['object_id'] not in unseen_oids]
            unseen_outputs = [o for o in outputs if o['object_id'] in unseen_oids]
            seen_outputs_all += seen_outputs
            unseen_outputs_all += unseen_outputs

            seen_loss = torch.mean(torch.Tensor([o['val_loss'] for o in seen_outputs]))
            unseen_loss = torch.mean(torch.Tensor([o['val_loss'] for o in unseen_outputs]))
            avg_loss = torch.mean(torch.Tensor([o['val_loss'] for o in outputs]))

            seen_err = torch.mean(torch.Tensor([o['val_err'] for o in seen_outputs]))
            unseen_err = torch.mean(torch.Tensor([o['val_err'] for o in unseen_outputs]))
            avg_err = torch.mean(torch.Tensor([o['val_err'] for o in outputs]))

            seen_err_wo_oracle = torch.mean(torch.Tensor([o['val_err_wo_oracle'] for o in seen_outputs]))
            unseen_err_wo_oracle = torch.mean(torch.Tensor([o['val_err_wo_oracle'] for o in unseen_outputs]))
            avg_err_wo_oracle = torch.mean(torch.Tensor([o['val_err_wo_oracle'] for o in outputs]))

            seen_add01d = self.add01d_metrics[dataset_name](map(lambda x: x['val_err_wo_oracle'], seen_outputs), map(lambda x: x['object_id'], seen_outputs))
            unseen_add01d = self.add01d_metrics[dataset_name](map(lambda x: x['val_err_wo_oracle'], unseen_outputs), map(lambda x: x['object_id'], unseen_outputs))
            avg_add01d = self.add01d_metrics[dataset_name](map(lambda x: x['val_err_wo_oracle'], outputs), map(lambda x: x['object_id'], outputs))

            log.update({
                ("val_err_%s" % dataset_name): avg_err, ("val_loss_%s" % dataset_name): avg_loss,
                ("val_seen_err_%s" % dataset_name): seen_err, ("val_unseen_err_%s" % dataset_name): unseen_err,
                ("val_add01d_%s" % dataset_name): avg_add01d,
                ("val_seen_add01d_%s" % dataset_name): seen_add01d, ("val_unseen_add01d_%s" % dataset_name): unseen_add01d,
            })

        '''Average over all datasets'''
        seen_loss = torch.mean(torch.Tensor([o['val_loss'] for o in seen_outputs_all]))
        unseen_loss = torch.mean(torch.Tensor([o['val_loss'] for o in unseen_outputs_all]))
        avg_loss = torch.mean(torch.Tensor([o['val_loss'] for o in outputs_all]))

        seen_err = torch.mean(torch.Tensor([o['val_err'] for o in seen_outputs_all]))
        unseen_err = torch.mean(torch.Tensor([o['val_err'] for o in unseen_outputs_all]))
        avg_err = torch.mean(torch.Tensor([o['val_err'] for o in outputs_all]))

        seen_err_wo_oracle = torch.mean(torch.Tensor([o['val_err_wo_oracle'] for o in seen_outputs_all]))
        unseen_err_wo_oracle = torch.mean(torch.Tensor([o['val_err_wo_oracle'] for o in unseen_outputs_all]))
        avg_err_wo_oracle = torch.mean(torch.Tensor([o['val_err_wo_oracle'] for o in outputs_all]))

        seen_add01d = self.add01d_metrics[dataset_name](map(lambda x: x['val_err_wo_oracle'], seen_outputs_all), map(lambda x: x['object_id'], seen_outputs_all))
        unseen_add01d = self.add01d_metrics[dataset_name](map(lambda x: x['val_err_wo_oracle'], unseen_outputs_all), map(lambda x: x['object_id'], unseen_outputs_all))
        avg_add01d = self.add01d_metrics[dataset_name](map(lambda x: x['val_err_wo_oracle'], outputs_all), map(lambda x: x['object_id'], outputs_all))

        log.update({"val_err": avg_err, "val_loss": avg_loss, "val_seen_err": seen_err, "val_unseen_err": unseen_err})
        log.update({"val_add01d": avg_add01d, "val_seen_add01d": seen_add01d, "val_unseen_add01d": unseen_add01d})

        to_return = {
            "log": log, "val_epoch_err": avg_err,
            "seen_err": seen_err,
            "unseen_err": unseen_err,
            "avg_err": avg_err,
            "seen_err_wo_oracle": seen_err_wo_oracle,
            "unseen_err_wo_oracle": unseen_err_wo_oracle,
            "avg_err_wo_oracle": avg_err_wo_oracle,
        }

        return to_return

    def test_step(self, batch, batch_idx):
        return self.validation_step(batch, batch_idx)

    def test_step_end(self, outputs):
        return self.validation_step_end(outputs)

    def test_epoch_end(self, outputs):
        '''Save the results'''
        # input_file = self.args.exp_name + "_" + self.args.resume_path.split('/')[-1].split("-")[0] + "_%s.csv" % self.args.data_split
        input_file = self.args.exp_name + "_%s.csv" % self.args.data_split
        save_path = os.path.join("test_logs", input_file)

        for d in ["test_logs", "test_logs/bop_results"]:
            if not os.path.exists(d):
                os.makedirs(d)

        saveOutputs(outputs, save_path)
        # with open(save_path, 'w', newline='') as csvfile:
        #     fieldnames = outputs[0].keys()
        #     writer = csv.DictWriter(csvfile, fieldnames=fieldnames, quoting = csv.QUOTE_NONE)
        #
        #     writer.writeheader()
        #     for i in range(len(outputs)):
        #         o = copy.deepcopy(outputs[i])
        #         # for k, v in o.items():
        #         #     if torch.is_tensor(v):
        #         #         print(k, v.shape)
        #         o = {k: to_np(v) for k, v in o.items()}
        #         o['mat'] = " ".join(["%0.6f"%_ for _ in o['mat'].reshape(-1).tolist()])
        #         o['mat_wo_oracle'] = " ".join(["%0.4f"%_ for _ in o['mat_wo_oracle'].reshape(-1).tolist()])
        #         writer.writerow(o)

        '''Again save the results in BOP format'''
        convertResultsBOP(
            input_root="test_logs",
            output_root=os.path.join("test_logs", "bop_results"),
            input_file=input_file,
            dataset_name=self.args.dataset_name[0],
            subname=None,
            mat_name="mat_wo_oracle"
        )
        convertResultsBOP(
            input_root="test_logs",
            output_root=os.path.join("test_logs", "bop_results"),
            input_file=input_file,
            dataset_name=self.args.dataset_name[0],
            subname="orahypo",
            mat_name="mat"
        )

        results = self.validation_epoch_end(outputs)
        print("w/ oracle hypo: avg %.3f, seen %.3f, unseen %.3f" % (
            results['avg_err'], results['seen_err'], results['unseen_err']
        ))

        print("w/o oracle hypo: avg %.3f, seen %.3f, unseen %.3f" % (
            results['avg_err_wo_oracle'], results['seen_err_wo_oracle'], results['unseen_err_wo_oracle']
        ))
        return results

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
        scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[30, 80], gamma=0.1)
        return [optimizer], [scheduler]


def to_np(x):
    if type(x) is int:
        return x
    elif torch.is_tensor(x):
        return x.detach().cpu().numpy()
    else:
        return x.detach().data.cpu().numpy()
