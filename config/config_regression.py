# -*- coding: utf-8 -*-
"""
@CreateTime :       2025/05/28 22:25
@File       :       config_regression.py
@Software   :       PyCharm
@Framework  :       Pytorch
@LastModify :       2025/07/30 23:35
"""
import os
import argparse
from utils.functions import Storage
from easydict import EasyDict as edict

class ConfigRegression():
    def __init__(self, args):
        self.globalArgs = args
        # hyper parameters for models
        HYPER_MODEL_MAP = {
            # 'mult': self.__MULT,
            # 'tfr_net': self.__TFR_Net,
            'hgatt_net': self.__HGAtt_Net,
            'gqa_net': self.__GQA_Net,
            
        }
        # hyper parameters for datasets
        HYPER_DATASET_MAP = self.__datasetCommonParams()

        # normalize
        model_name = str.lower(args.modelName)
        dataset_name = str.lower(args.datasetName)
        # load params
        commonArgs = HYPER_MODEL_MAP[model_name]()['commonParas']
        dataArgs = HYPER_DATASET_MAP[dataset_name]

        if commonArgs['data_missing']:
            dataArgs = dataArgs['aligned_missing'] if (args.need_data_aligned and 'aligned_missing' in dataArgs) else dataArgs['unaligned_missing']
        else:
            dataArgs = dataArgs['aligned'] if (args.need_data_aligned and 'aligned' in dataArgs) else dataArgs['unaligned']

        self.args = edict(dict(vars(args),
                                 **dataArgs,
                                 **commonArgs,
                                 **HYPER_MODEL_MAP[model_name]()['datasetParas'][dataset_name],
                                 ))

    def __datasetCommonParams(self):
        from config.get_data_root import data_root
        # NOTE: change the dataset_path to your own path!
        root_dataset_dir = data_root
        tmp = {
            'mosi':{
                'aligned': {
                    'dataPath': os.path.join(root_dataset_dir, 'MOSI/Processed/aligned_50.pkl'),
                    # 'dataPath': 'data/MOSI/Processed/aligned_50.pkl',
                    'seq_lens': (50, 50, 50),
                    # (text, audio, video)
                    'feature_dims': (768, 5, 20),  # (768, 74, 35),
                    'train_samples': 1284,
                    'num_classes': 3,
                    'language': 'en',
                    'KeyEval': 'Loss' 
                },
                'unaligned': {
                    'dataPath': os.path.join(root_dataset_dir, 'MOSI/Processed/unaligned_50.pkl'),
                    # 'dataPath': 'data/MOSI/Processed/unaligned_50.pkl',
                    'seq_lens': None,  # None,
                    # (text, audio, video)
                    'feature_dims':  (768, 5, 20),  # (768, 74, 35),
                    'train_samples': 1284,
                    'num_classes': 3,
                    'language': 'en',
                    'KeyEval': 'Loss' 
                },
                'aligned_missing': {
                    'dataPath': os.path.join(root_dataset_dir, 'MOSI/Processed/aligned_50.pkl'),
                    # 'dataPath': 'data/MOSI/Processed/aligned_50.pkl',
                    'seq_lens': (50, 50, 50),
                    'feature_dims': (768, 5, 20),  # (768, 74, 35),
                    'train_samples': 1284,
                    'num_classes': 3,
                    'language': 'en',
                    'KeyEval': 'Loss',
                    # 'missing_rate': (0.0, 0.0, 0.0),
                    'missing_seed': (111, 1111, 11111),
                },
                'unaligned_missing': {
                    'dataPath': os.path.join(root_dataset_dir, 'MOSI/Processed/unaligned_50.pkl'),
                    # 'dataPath': 'data/MOSI/Processed/unaligned_50.pkl',
                    'seq_lens': None,  # None,
                    'feature_dims': (768, 5, 20),  # (768, 74, 35),
                    'train_samples': 1284,
                    'num_classes': 3,
                    'language': 'en',
                    'KeyEval': 'Loss',
                    # 'missing_rate': (0.1, 0.1, 0.1),
                    'missing_seed': (111, 1111, 11111),
                }
            },
            'mosei': {
                'aligned': {
                    'dataPath': os.path.join(root_dataset_dir, 'MOSEI/Processed/aligned_50.pkl'),
                    # 'dataPath': os.path.join(root_dataset_dir, 'MOSEI/aligned_50.pkl'),
                    # 'dataPath': 'data/MOSEI/aligned_50.pkl',
                    'seq_lens': (50, 50, 50),
                    # (text, audio, video)
                    'feature_dims': (768, 74, 35),
                    'train_samples': 16326,
                    'num_classes': 3,
                    'language': 'en',
                    'KeyEval': 'Loss'
                },
                'unaligned': {
                    'dataPath': os.path.join(root_dataset_dir, 'MOSEI/Processed/unaligned_50.pkl'),
                    # 'dataPath': 'data/MOSEI/unaligned_50.pkl',
                    'seq_lens': (50, 500, 375),
                    # 'seq_lens': (50, 50, 50),
                    # (text, audio, video)
                    'feature_dims': (768, 74, 35),
                    'train_samples': 16326,
                    'num_classes': 3,
                    'language': 'en',
                    'KeyEval': 'Loss'
                },
                'aligned_missing': {
                    'dataPath': os.path.join(root_dataset_dir, 'MOSEI/Processed/aligned_50.pkl'),
                    # 'dataPath': 'data/MOSEI/aligned_50.pkl',
                    'seq_lens': (50, 50, 50),
                    # (text, audio, video)
                    'feature_dims': (768, 74, 35),
                    'train_samples': 16326,
                    'num_classes': 3,
                    'language': 'en',
                    'KeyEval': 'Loss',
                    # 'missing_rate': (0.0, 0.0, 0.0),
                    'missing_seed': (111, 1111, 11111),
                },
                'unaligned_missing': {
                    'dataPath': os.path.join(root_dataset_dir, 'MOSEI/Processed/unaligned_50.pkl'),
                    # 'dataPath': 'data/MOSEI/unaligned_50.pkl',
                    'seq_lens': (50, 500, 375),
                    # 'seq_lens': (50, 50, 50),
                    # (text, audio, video)
                    'feature_dims': (768, 74, 35),
                    'train_samples': 16326,
                    'num_classes': 3,
                    'language': 'en',
                    'KeyEval': 'Loss',
                    # 'missing_rate': (0.1, 0.1, 0.1),
                    'missing_seed': (111, 1111, 11111),
                }
            },
            'sims':{
                'unaligned': {
                    'dataPath': os.path.join(root_dataset_dir, 'SIMS/Processed/unaligned_39.pkl'),
                    # 'dataPath': os.path.join(root_dataset_dir, f'SIMS/Processed/unaligned_39{"" if not self.globalArgs.use_normalized_data else "_normalized"}.pkl'),
                    # (batch_size, seq_lens, feature_dim)
                    'seq_lens': (50, 400, 55), # (text, audio, video)
                    'feature_dims': (768, 33, 709), # (text, audio, video)
                    'train_samples': 1368,
                    'num_classes': 3,
                    'language': 'cn',
                    'KeyEval': 'Loss',
                },
                'unaligned_missing': {
                    'dataPath': os.path.join(root_dataset_dir, 'SIMS/Processed/unaligned_39.pkl'),
                    # 'dataPath': os.path.join(root_dataset_dir, f'SIMS/Processed/unaligned_39{"" if not self.globalArgs.use_normalized_data else "_normalized"}.pkl'),
                    'seq_lens': None,
                    'feature_dims': (768, 33, 709), # (text, audio, video)
                    'train_samples': 1368,
                    'num_classes': 3,
                    'language': 'cn',
                    'KeyEval': 'Loss',
                    'missing_seed': (111, 1111, 11111),
                }
            }
        }
        return tmp

    def __HGAtt_Net(self):
        tmp = {
            'commonParas':{
                'data_missing': True,  # True
                'deal_missing': False,
                # 'need_data_aligned': True,  # False --> unaligned data ; True --> aligned data
                # 'need_model_aligned': False, # False --> complete modality setting ; True --> incomplete modality setting
                'need_normalized': False,
                'use_bert': True,  # True,
                'use_finetune': True,
                'save_labels': False,
                # 'early_stop': 100,
                'update_epochs': 5
            },
            # dataset
            'datasetParas':{
                'mosi':{
                    # the batch_size of each epoch is update_epochs * batch_size
                    # 'batch_size': 8,
                    'learning_rate_bert': 2e-5,  # 5e-5
                    'learning_rate_audio': 1e-4,  #  1e-4
                    'learning_rate_video': 1e-4,
                    'learning_rate_other': 1e-4,  #  1e-4
                    'weight_decay_bert': 0.001,
                    'weight_decay_audio': 0.01,
                    'weight_decay_video': 0.001,
                    'weight_decay_other': 0.001,
                    # feature subNets
                    'a_lstm_hidden_size': 32,
                    'v_lstm_hidden_size': 64,
                    'a_lstm_layers': 1,
                    'v_lstm_layers': 1,
                    'text_out': 768,
                    'audio_out': 16,
                    'video_out': 32,
                    'a_lstm_dropout': 0.0,
                    'v_lstm_dropout': 0.0,
                    't_bert_dropout':0.1,
                    # fusion
                    'd_model': 128,
                    'fusion_layers': 3,
                    'heads': 4,
                    'learnable_pos_emb': False,
                    'emb_dropout': 0.0,
                    'attn_dropout': 0.3,
                    'ff_dropout': 0.1,
                    'ff_expansion': 4,
                    'mpu_share': True,
                    'modality_share': True,
                    'layer_share': True,
                    'attn_act_fn': 'tanh',
                    # predictor
                    'gmc_tokens_pred_dim': 128,
                    'text_pred_dim': 256,
                    'audio_pred_dim': 8,
                    'video_pred_dim': 16,
                    # loss
                    'recon_loss': 'SmoothL1Loss',
                    'loss_attra_weight': 1, # high-level feature attraction
                    'loss_recon_weight': 1, # low-level feature reconstruction
                    # post feature
                    'post_fusion_dim': 256,
                    'post_fusion_dropout': 0.3,
                    # res
                    'H': 3.0
                },
                'mosei':{
                    # the batch_size of each epoch is update_epochs * batch_size
                    # 'batch_size': 32,
                    'learning_rate_bert': 2e-5,  # 2e-5
                    'learning_rate_audio': 1e-4,  # (原来的）1e-4 -> 2e-5 -> 1e-5
                    'learning_rate_video': 1e-4,  # (原来的）1e-4 -> 2e-5 -> 1e-5
                    'learning_rate_other': 1e-4,  # (原来的）1e-4 -> 2e-5 -> 1e-5
                    'weight_decay_bert': 0.001,
                    'weight_decay_audio': 0.001,
                    'weight_decay_video': 0.001,
                    'weight_decay_other': 0.001,
                    # feature subNets
                    'a_lstm_hidden_size': 16,
                    'v_lstm_hidden_size': 32,
                    'a_lstm_layers': 1,
                    'v_lstm_layers': 1,
                    'text_out': 768,
                    'audio_out': 16,
                    'video_out': 32,
                    'a_lstm_dropout': 0.0,
                    'v_lstm_dropout': 0.0,
                    't_bert_dropout':0.1,
                    # fusion
                    'd_model': 128,
                    'fusion_layers': 2,
                    'heads': 4,
                    'learnable_pos_emb': False,
                    'emb_dropout': 0.0,
                    'attn_dropout': 0.0,
                    'ff_dropout': 0.0,
                    'ff_expansion': 4,
                    'mpu_share': True,
                    'modality_share': True,
                    'layer_share': True,
                    'attn_act_fn': 'tanh',
                    # predictor
                    'gmc_tokens_pred_dim': 128,
                    'text_pred_dim': 256,
                    'audio_pred_dim': 8,
                    'video_pred_dim': 16,
                    # loss
                    'recon_loss': 'SmoothL1Loss',
                    'loss_attra_weight': 1, # high-level feature attraction
                    'loss_recon_weight': 1, # low-level feature reconstruction
                    # post feature
                    'post_fusion_dim': 128,
                    'post_fusion_dropout': 0.0,
                    # res
                    'H': 3.0
                },
                'sims':{
                    # the batch_size of each epoch is update_epochs * batch_size
                    'batch_size': 16,
                    'learning_rate_bert': 2e-5,
                    'learning_rate_audio': 1e-4,  # 1e-3
                    'learning_rate_video': 1e-4,  # 1e-3
                    'learning_rate_other': 1e-4,  # 1e-3
                    'weight_decay_bert': 0.001,
                    'weight_decay_audio': 0.0,
                    'weight_decay_video': 0.0,
                    'weight_decay_other': 0.0,
                    # feature subNets
                    'a_lstm_hidden_size': 16,
                    'v_lstm_hidden_size': 32,
                    'a_lstm_layers': 1,
                    'v_lstm_layers': 1,
                    'text_out': 768,
                    'audio_out': 16,
                    'video_out': 32,
                    'a_lstm_dropout': 0.0,
                    'v_lstm_dropout': 0.0,
                    't_bert_dropout':0.1,
                    # fusion
                    'd_model': 32,
                    'fusion_layers': 4,
                    'heads': 4,
                    'learnable_pos_emb': False,
                    'emb_dropout': 0.0,
                    'attn_dropout': 0.0,
                    'ff_dropout': 0.0,
                    'ff_expansion': 4,
                    'mpu_share': True,
                    'modality_share': True,
                    'layer_share': True,
                    'attn_act_fn': 'tanh',
                    # predictor
                    'gmc_tokens_pred_dim': 128,
                    'text_pred_dim': 256,
                    'audio_pred_dim': 8,
                    'video_pred_dim': 16,
                    # loss
                    'recon_loss': 'SmoothL1Loss',
                    'loss_attra_weight': 0.5, # high-level feature attraction
                    'loss_recon_weight': 0.5, # low-level feature reconstruction
                    # post feature
                    'post_fusion_dim': 256,
                    'post_fusion_dropout': 0.0,
                    # res
                    'H': 1.0
                },
            },
        }
        return tmp

    def __GQA_Net(self):
        tmp = {
            'commonParas':{
                'data_missing': True,  # True
                'deal_missing': False,
                # 'need_data_aligned': True,  # False --> unaligned data ; True --> aligned data
                # 'need_model_aligned': False, # False --> complete modality setting ; True --> incomplete modality setting
                'need_normalized': False,
                'use_bert': True,  # True,
                'use_finetune': True,
                'save_labels': False,
                # 'early_stop': 100,
                'update_epochs': 5
            },
            # dataset
            'datasetParas':{
                'mosi':{
                    # the batch_size of each epoch is update_epochs * batch_size
                    # 'batch_size': 8,
                    'learning_rate_bert': 2e-5,  # 5e-5
                    'learning_rate_audio': 1e-4,  #  1e-4
                    'learning_rate_video': 1e-4,
                    'learning_rate_other': 1e-4,  #  1e-4
                    'weight_decay_bert': 0.001,
                    'weight_decay_audio': 0.01,
                    'weight_decay_video': 0.001,
                    'weight_decay_other': 0.001,
                    # feature subNets
                    'a_lstm_hidden_size': 32,
                    'v_lstm_hidden_size': 64,
                    'a_lstm_layers': 1,
                    'v_lstm_layers': 1,
                    'text_out': 768,
                    'audio_out': 16,
                    'video_out': 32,
                    'a_lstm_dropout': 0.0,
                    'v_lstm_dropout': 0.0,
                    't_bert_dropout':0.1,
                    # fusion
                    'd_model': 128,
                    'fusion_layers': 3,
                    'heads': 4,
                    'learnable_pos_emb': False,
                    'emb_dropout': 0.0,
                    'attn_dropout': 0.3,
                    'ff_dropout': 0.1,
                    'ff_expansion': 4,
                    'mpu_share': True,
                    'modality_share': True,
                    'layer_share': True,
                    'attn_act_fn': 'tanh',
                    # predictor
                    'gmc_tokens_pred_dim': 128,
                    'text_pred_dim': 256,
                    'audio_pred_dim': 8,
                    'video_pred_dim': 16,
                    # loss
                    'recon_loss': 'SmoothL1Loss',
                    'loss_attra_weight': 1, # high-level feature attraction
                    'loss_recon_weight': 1, # low-level feature reconstruction
                    # post feature
                    'post_fusion_dim': 256,
                    'post_fusion_dropout': 0.3,
                    # res
                    'H': 3.0
                },
                'mosei':{
                    # the batch_size of each epoch is update_epochs * batch_size
                    # 'batch_size': 32,
                    'learning_rate_bert': 2e-5,  # 2e-5
                    'learning_rate_audio': 1e-4,  # (原来的）1e-4 -> 2e-5 -> 1e-5
                    'learning_rate_video': 1e-4,  # (原来的）1e-4 -> 2e-5 -> 1e-5
                    'learning_rate_other': 1e-4,  # (原来的）1e-4 -> 2e-5 -> 1e-5
                    'weight_decay_bert': 0.001,
                    'weight_decay_audio': 0.001,
                    'weight_decay_video': 0.001,
                    'weight_decay_other': 0.001,
                    # feature subNets
                    'a_lstm_hidden_size': 16,
                    'v_lstm_hidden_size': 32,
                    'a_lstm_layers': 1,
                    'v_lstm_layers': 1,
                    'text_out': 768,
                    'audio_out': 16,
                    'video_out': 32,
                    'a_lstm_dropout': 0.0,
                    'v_lstm_dropout': 0.0,
                    't_bert_dropout':0.1,
                    # fusion
                    'd_model': 128,
                    'fusion_layers': 2,
                    'heads': 4,
                    'learnable_pos_emb': False,
                    'emb_dropout': 0.0,
                    'attn_dropout': 0.0,
                    'ff_dropout': 0.0,
                    'ff_expansion': 4,
                    'mpu_share': True,
                    'modality_share': True,
                    'layer_share': True,
                    'attn_act_fn': 'tanh',
                    # predictor
                    'gmc_tokens_pred_dim': 128,
                    'text_pred_dim': 256,
                    'audio_pred_dim': 8,
                    'video_pred_dim': 16,
                    # loss
                    'recon_loss': 'SmoothL1Loss',
                    'loss_attra_weight': 1, # high-level feature attraction
                    'loss_recon_weight': 1, # low-level feature reconstruction
                    # post feature
                    'post_fusion_dim': 128,
                    'post_fusion_dropout': 0.0,
                    # res
                    'H': 3.0
                },
                'sims':{
                    # the batch_size of each epoch is update_epochs * batch_size
                    'batch_size': 16,
                    'learning_rate_bert': 2e-5,
                    'learning_rate_audio': 1e-4,  # 1e-3
                    'learning_rate_video': 1e-4,  # 1e-3
                    'learning_rate_other': 1e-4,  # 1e-3
                    'weight_decay_bert': 0.001,
                    'weight_decay_audio': 0.0,
                    'weight_decay_video': 0.0,
                    'weight_decay_other': 0.0,
                    # feature subNets
                    'a_lstm_hidden_size': 16,
                    'v_lstm_hidden_size': 32,
                    'a_lstm_layers': 1,
                    'v_lstm_layers': 1,
                    'text_out': 768,
                    'audio_out': 16,
                    'video_out': 32,
                    'a_lstm_dropout': 0.0,
                    'v_lstm_dropout': 0.0,
                    't_bert_dropout':0.1,
                    # fusion
                    'd_model': 32,
                    'fusion_layers': 4,
                    'heads': 4,
                    'learnable_pos_emb': False,
                    'emb_dropout': 0.0,
                    'attn_dropout': 0.0,
                    'ff_dropout': 0.0,
                    'ff_expansion': 4,
                    'mpu_share': True,
                    'modality_share': True,
                    'layer_share': True,
                    'attn_act_fn': 'tanh',
                    # predictor
                    'gmc_tokens_pred_dim': 128,
                    'text_pred_dim': 256,
                    'audio_pred_dim': 8,
                    'video_pred_dim': 16,
                    # loss
                    'recon_loss': 'SmoothL1Loss',
                    'loss_attra_weight': 0.5, # high-level feature attraction
                    'loss_recon_weight': 0.5, # low-level feature reconstruction
                    # post feature
                    'post_fusion_dim': 256,
                    'post_fusion_dropout': 0.0,
                    # res
                    'H': 1.0
                },
            },
        }
        return tmp

    def get_config(self):
        return self.args
