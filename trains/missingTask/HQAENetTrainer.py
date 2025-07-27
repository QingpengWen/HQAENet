# -*- coding: utf-8 -*-
"""
@CreateTime :       2025/06/18 08:18
@File       :       GQATrainer.py
@Software   :       PyCharm
@Framework  :       Pytorch
@LastModify :       2025/07/30 23:35
"""
import gc
import os
import re
import sys
import time
import logging
import math
import copy
import argparse
import numpy as np
import pickle as plk
from glob import glob
import pandas as pd
import torch.nn.functional as F
from tqdm import tqdm
import time

import torch
import torch.nn as nn
from torch import optim

from cross_datasets import load_test_loader
from utils.ContrastStudy import SupConLoss
from utils.functions import dict_to_str
from utils.metricsTop import MetricsTop

logger = logging.getLogger('MSA')

class HQAENetTrainer():
    def __init__(self, args):
        assert args.train_mode == 'regression'

        self.args = args
        self.args.tasks = "M"
        self.metrics = MetricsTop(args.train_mode).getMetics(args.datasetName)
        self.__SupConLoss = SupConLoss()

    def do_train(self, model, dataloader):

        print("Thank you for recognizing our work. \n"
              "Considering the protection of the originality of our work, "
              "the training code will be provided after acceptance!")

        sys.exit(0)

    def do_test(self, model, dataloader, criterion_attra=None, criterion_recon=None, mode="VAL", epochs=None, batch=None, do_test=None, cross_eval=False):
        if epochs is None:
            logger.info("=" * 30 + f"Start Test of Seed {self.args.seed}" + "=" * 30)
        model.eval()
        y_pred, y_true = [], []
        eval_loss = 0.0
        eval_loss_pred = 0.0
        if criterion_attra is None: criterion_attra = nn.CosineSimilarity(dim=1).cuda(self.args.gpu_ids)
        if criterion_recon is None: criterion_recon = ReconLoss(self.args.recon_loss)
        with torch.no_grad():
            with tqdm(dataloader) as td:
                for batch_idx, batch_data in enumerate(td, 1):
                    # complete view
                    vision = batch_data['vision'].to(self.args.device)
                    audio = batch_data['audio'].to(self.args.device)
                    text = batch_data['text'].to(self.args.device)
                    # incomplete (missing) view
                    vision_m = batch_data['vision_m'].to(self.args.device)
                    audio_m = batch_data['audio_m'].to(self.args.device)
                    text_m = batch_data['text_m'].to(self.args.device)
                    vision_missing_mask = batch_data['vision_missing_mask'].to(self.args.device)
                    audio_missing_mask = batch_data['audio_missing_mask'].to(self.args.device)
                    text_missing_mask = batch_data['text_missing_mask'].to(self.args.device)
                    vision_mask = batch_data['vision_mask'].to(self.args.device)
                    audio_mask = batch_data['audio_mask'].to(self.args.device)

                    audio_lengths = batch_data['audio_lengths'].to(self.args.device)
                    vision_lengths = batch_data['vision_lengths'].to(self.args.device)

                    labels = batch_data['labels']['M'].to(self.args.device).view(-1)
                    # Add feature transfer: MOSI->MOSEI
                    # if self.args.datasetName == "mosi":
                    #     if cross_eval != True:
                    #         # if self.args.do_evaluation:
                    #         audio = self.a_linear(audio)
                    #         audio_m = self.a_linear(audio_m)
                    #         vision = self.v_linear(vision)
                    #         vision_m = self.v_linear(vision_m)
                    if mode == "VAL":
                        outputs = model((text, text_m), (audio, audio_m, audio_lengths),
                                        (vision, vision_m, vision_lengths), do_test=do_test)
                    elif mode == "TEST":
                        outputs = model((text, text_m), (audio, audio_m, audio_lengths),
                                        (vision, vision_m, vision_lengths), do_test=do_test)
                    else:
                        outputs = model((text, text_m), (audio, audio_m, audio_lengths), (vision, vision_m, vision_lengths), do_test=True)

                    # compute loss
                    loss_pred_m = torch.mean(torch.abs(outputs['pred_m'].view(-1) - labels.view(-1)))
                    loss_pred = torch.mean(torch.abs(outputs['pred'].view(-1) - labels.view(-1)))  # prediction loss of complete view
                    loss_predict = loss_pred_m + loss_pred
                    loss_attra_gmc_tokens = -(criterion_attra(outputs['p_gmc_tokens_m'], outputs['z_gmc_tokens']).mean() +
                                          criterion_attra(outputs['p_gmc_tokens'], outputs['z_gmc_tokens_m']).mean()) * 0.5
                    loss_attra_text = -(criterion_attra(outputs['p_text_m'], outputs['z_text']).mean() +
                                          criterion_attra(outputs['p_text'], outputs['z_text_m']).mean()) * 0.5
                    loss_attra_audio = -(criterion_attra(outputs['p_audio_m'], outputs['z_audio']).mean() +
                                          criterion_attra(outputs['p_audio'], outputs['z_audio_m']).mean()) * 0.5
                    loss_attra_video = -(criterion_attra(outputs['p_video_m'], outputs['z_video']).mean() +
                                          criterion_attra(outputs['p_video'], outputs['z_video_m']).mean()) * 0.5                
                    loss_attra = loss_attra_gmc_tokens + loss_attra_text + loss_attra_audio + loss_attra_video
                    mask = text[:, 1, :] - text_missing_mask[:, :] # '1:' for excluding CLS
                    loss_recon_text = criterion_recon(outputs['text_recon'], outputs['text_for_recon'], mask)
                    max_length = 50
                    batch_number = audio_lengths.numel()
                    lengths = torch.linspace(3, max_length, steps=batch_number)
                    mask = audio_mask - audio_missing_mask
                    loss_recon_audio = criterion_recon(outputs['audio_recon'], audio[:, :  int(lengths.max())], mask[:, : int(lengths.max())])
                    mask = vision_mask - vision_missing_mask
                    loss_recon_video = criterion_recon(outputs['video_recon'], vision[:, : int(lengths.max())], mask[:, : int(lengths.max())])
                    loss_recon = loss_recon_text + loss_recon_audio + loss_recon_video
                    loss = loss_predict + self.args.loss_attra_weight * loss_attra + self.args.loss_recon_weight * loss_recon
                    eval_loss += loss.item()
                    eval_loss_pred += loss_pred_m.item()
                    y_pred.append(outputs['pred'].cpu())
                    y_true.append(labels.cpu())

        eval_loss = eval_loss / len(dataloader)
        eval_loss_pred = eval_loss_pred / len(dataloader)
        pred, true = torch.cat(y_pred), torch.cat(y_true)
        eval_results = self.metrics(pred, true)
        eval_results['Loss'] = eval_loss
        eval_results['Loss(pred_m)'] = eval_loss_pred
        if epochs is None:  # for TEST
            logger.info(f"\n [Test] {dict_to_str(eval_results)}")
            logger.info(f"==> Note: achieve this results (best [Val]) / {getattr(self, 'best_test_epoch', None)} (best [Test])")
        return eval_results


class ReconLoss(nn.Module):
    def __init__(self, type):
        super().__init__()
        self.eps = 1e-6
        self.type = type
        if type == 'L1Loss':
            self.loss = nn.L1Loss(reduction='sum')
        elif type == 'SmoothL1Loss':
            self.loss = nn.SmoothL1Loss(reduction='sum')
        elif type == 'MSELoss':
            self.loss = nn.MSELoss(reduction='sum')
        else:
            raise NotImplementedError

    def forward(self, pred, target, mask):
        """
            pred, target -> batch, seq_len, d
            mask -> batch, seq_len
        """
        max_length = 50
        if int(mask.shape[1]) < max_length:
            padding_length = max_length - mask.shape[1]
            padding = torch.zeros(mask.shape[0], padding_length, dtype=mask.dtype, device=mask.device)
            mask = torch.cat((mask, padding), dim=1)
        mask = mask.unsqueeze(-1).expand(pred.shape[0], pred.shape[1], pred.shape[2]).float()

        loss = self.loss(pred*mask, target*mask) / (torch.sum(mask) + self.eps)

        return loss

def angular_margin_loss(feature, label, weight, scale_factor=30.0, margin=0.5, lambda_l2=0.01):

    labels = [0, 1, 2]
    normalized_feature = F.normalize(feature, p=2, dim=0)
    normalized_weight = F.normalize(weight, p=2, dim=1)

    cos_theta = torch.matmul(normalized_feature, normalized_weight.t())  # 形状为 (num_classes,)

    correct_class_cos_theta = cos_theta[label]

    cos_theta_m = correct_class_cos_theta - margin

    cos_theta_with_margin = cos_theta.clone()
    cos_theta_with_margin[label] = cos_theta_m

    exp_cos_theta = torch.exp(scale_factor * cos_theta_with_margin)
    softmax_output = exp_cos_theta / exp_cos_theta.sum()

    loss = -torch.log(softmax_output[label])

    l2_reg = lambda_l2 * (weight ** 2).sum()

    total_loss = loss + l2_reg

    return total_loss

def discriminator_loss(real_output, fake_output):
    real_loss = F.binary_cross_entropy(real_output, torch.ones_like(real_output))

def get_domain_loss(self, model):

    W = model.Model.W.discriminator_layer_2.weight
    pred_shared_t = model.Model.D_ht
    pred_shared_a = model.Model.D_ha
    pred_shared_v = model.Model.D_hv

    Lami = torch.zeros(1, device="cuda")
    for i in range(pred_shared_v.size(0)):
        Lami += (angular_margin_loss(feature=pred_shared_t[i], label=0, weight=W) + angular_margin_loss(
            feature=pred_shared_a[i], label=1, weight=W) + angular_margin_loss(feature=pred_shared_v[i], label=2, weight=W)) / 3.0
    Lami = Lami / pred_shared_v.size(0)
    return Lami