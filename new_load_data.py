# -*- coding: utf-8 -*-
"""
@CreateTime :       2025/05/30 12:21
@File       :       load_data.py
@Software   :       PyCharm
@Framework  :       Pytorch
@LastModify :       2025/07/30 23:35
"""
import gc
import mmap
import os
import logging
import pickle
import numpy as np

import torch
from torch.utils.data import Dataset, DataLoader

__all__ = ['MMDataLoader']

logger = logging.getLogger('MSA')


class MMDataset(Dataset):
    def __init__(self, args, mode='train'):
        self.mode = mode
        self.args = args
        DATA_MAP = {
            'mosi': self.__init_mosi,
            'mosei': self.__init_mosei,
            'sims': self.__init_sims,
        }
        DATA_MAP[args.datasetName]()

    def __init_mosi(self):
        with open(self.args.dataPath, 'rb') as f:
            data = pickle.load(f)

        if self.args.use_bert:
            self.text = data[self.mode]['text_bert'].astype(np.float32)
        else:
            self.text = data[self.mode]['text'].astype(np.float32)
        self.vision = data[self.mode]['vision'].astype(np.float32)
        self.audio = data[self.mode]['audio'].astype(np.float32)
        self.rawText = data[self.mode]['raw_text']
        self.ids = data[self.mode]['id']

        self.labels = {
            'M': data[self.mode][self.args.train_mode + '_labels'].astype(np.float32)
        }
        if self.args.datasetName == 'sims':
            for m in "TAV":
                self.labels[m] = data[self.mode][self.args.train_mode + '_labels_' + m]

        logger.info(f"{self.mode} samples: {self.labels['M'].shape}")

        if not self.args.need_data_aligned:
            self.audio_lengths = data[self.mode]['audio_lengths']
            self.vision_lengths = data[self.mode]['vision_lengths']
        else:
            text_lengths = np.sum(self.text[:, 1], axis=1).astype(np.int16).tolist()
            self.audio_lengths, self.vision_lengths = text_lengths, text_lengths
        self.audio[self.audio == -np.inf] = 0

        if self.args.data_missing:
            # 只生成掩码，不生成处理后的数据
            self.text_missing_mask, self.text_mask, self.text_length = self.generate_mask(
                input_mask=self.text[:, 1, :] if self.args.use_bert else None,
                input_len=self.audio_lengths if not self.args.need_data_aligned and not self.args.use_bert else None,
                missing_rate=self.args.missing_rate[0],
                missing_seed=self.args.missing_seed[0],
                mode='text',
                max_seq_len=self.text.shape[2] if self.args.use_bert else self.text.shape[1]
            )
            self.audio_missing_mask, self.audio_mask, self.audio_length = self.generate_mask(
                input_mask=None,
                input_len=self.audio_lengths,
                missing_rate=self.args.missing_rate[1],
                missing_seed=self.args.missing_seed[1],
                mode='audio',
                max_seq_len=self.audio.shape[1]
            )
            self.vision_missing_mask, self.vision_mask, self.vision_length = self.generate_mask(
                input_mask=None,
                input_len=self.vision_lengths,
                missing_rate=self.args.missing_rate[2],
                missing_seed=self.args.missing_seed[2],
                mode='vision',
                max_seq_len=self.vision.shape[1]
            )

        if self.args.seq_lens != None:
            self.__truncated()
        if self.args.need_normalized:
            self.__normalize()

    def __init_mosei(self):
        return self.__init_mosi()

    def __init_sims(self):
        return self.__init_mosi()

    def generate_mask(self, input_mask, input_len, missing_rate, missing_seed, mode='text', max_seq_len=None):
        """只生成缺失掩码，不创建处理后的数据"""
        if mode == 'text':
            if input_mask is not None:
                # 将input_mask转换为布尔数组（原为0/1的浮点数）
                input_mask_arr = (input_mask > 0.5).astype(bool)
                # 计算有效长度：第一个0的位置，如果没有0，则取全长
                has_false = np.any(~input_mask_arr, axis=1)
                full_length = input_mask_arr.shape[1]
                # 初始化input_len为full_length
                input_len_arr = np.full(input_mask_arr.shape[0], full_length, dtype=int)
                # 对于有false的行，取第一个false的位置
                first_false_indices = np.argmax(~input_mask_arr, axis=1)
                input_len_arr[has_false] = first_false_indices[has_false]
            else:
                # 不使用BERT的情况（普通文本），这里我们暂不支持
                raise ValueError("In missing data mode for text, we only support BERT input. Set use_bert=True.")
        else:  # audio or vision
            # 确保input_len是整数列表
            input_len = [int(x) for x in input_len]
            input_mask_arr = np.zeros((len(input_len), max_seq_len), dtype=bool)
            for i, length in enumerate(input_len):
                if length > max_seq_len:
                    length = max_seq_len
                input_mask_arr[i, :length] = True
            input_len_arr = input_len  # 注意：这里input_len_arr就是传入的input_len

        np.random.seed(missing_seed)
        # 生成随机数矩阵，形状与input_mask_arr相同
        random_matrix = np.random.random(size=input_mask_arr.shape)
        # 生成缺失掩码：保留的概率为1-missing_rate，并且只保留input_mask_arr为True的区域
        missing_mask = (random_matrix > missing_rate) & input_mask_arr

        if mode == 'text':
            # 确保[CLS]（位置0）和[SEP]（位置input_len_arr[i]-1）不被掩码
            for i in range(missing_mask.shape[0]):
                # [CLS] always at 0
                missing_mask[i, 0] = True
                if input_len_arr[i] > 1:
                    sep_index = input_len_arr[i] - 1
                    if sep_index < max_seq_len:
                        missing_mask[i, sep_index] = True

        return missing_mask, input_mask_arr, input_len_arr

    def __truncated(self):
        # 保持不变
        pass

    def __normalize(self):
        # 移除对处理数据的归一化
        self.vision = np.transpose(self.vision, (1, 0, 2))
        self.audio = np.transpose(self.audio, (1, 0, 2))
        self.vision = np.mean(self.vision, axis=0, keepdims=True)
        self.audio = np.mean(self.audio, axis=0, keepdims=True)
        self.vision[self.vision != self.vision] = 0
        self.audio[self.audio != self.audio] = 0
        self.vision = np.transpose(self.vision, (1, 0, 2))
        self.audio = np.transpose(self.audio, (1, 0, 2))

    def __len__(self):
        return len(self.labels['M'])

    def get_seq_len(self):
        if self.args.use_bert:
            return (self.text.shape[2], self.audio.shape[1], self.vision.shape[1])
        else:
            return (self.text.shape[1], self.audio.shape[1], self.vision.shape[1])

    def get_feature_dim(self):
        return self.text.shape[2], self.audio.shape[2], self.vision.shape[2]

    def __getitem__(self, index):
        if self.args.data_missing:
            # 实时生成处理后的数据
            if self.args.use_bert:
                # 处理BERT文本数据
                text_sample = self.text[index]
                input_ids = text_sample[0]
                text_m = np.zeros_like(input_ids)
                valid_indices = self.text_missing_mask[index]
                text_m[valid_indices] = input_ids[valid_indices]
                text_m[~valid_indices & self.text_mask[index]] = 100  # UNK token

                # 重组文本数据
                text_m_sample = np.array([
                    text_m,
                    text_sample[1],
                    text_sample[2]
                ])
            else:
                # 处理普通文本数据
                text_sample = self.text[index]
                text_m_sample = text_sample.copy()
                valid_indices = self.text_missing_mask[index]
                text_m_sample[~valid_indices] = 100  # UNK token

            # 处理音频数据
            audio_sample = self.audio[index]
            audio_m_sample = audio_sample.copy()
            audio_m_sample[~self.audio_missing_mask[index]] = 0

            # 处理视觉数据
            vision_sample = self.vision[index]
            vision_m_sample = vision_sample.copy()
            vision_m_sample[~self.vision_missing_mask[index]] = 0

            sample = {
                'text': torch.Tensor(self.text[index]),
                'text_m': torch.Tensor(text_m_sample),
                'text_missing_mask': torch.Tensor(self.text_missing_mask[index].astype(float)),
                'audio': torch.Tensor(audio_sample),
                'audio_m': torch.Tensor(audio_m_sample),
                'audio_lengths': self.audio_lengths[index],
                'audio_mask': torch.Tensor(self.audio_mask[index].astype(float)),
                'audio_missing_mask': torch.Tensor(self.audio_missing_mask[index].astype(float)),
                'vision': torch.Tensor(vision_sample),
                'vision_m': torch.Tensor(vision_m_sample),
                'vision_lengths': self.vision_lengths[index],
                'vision_mask': torch.Tensor(self.vision_mask[index].astype(float)),
                'vision_missing_mask': torch.Tensor(self.vision_missing_mask[index].astype(float)),
                'index': index,
                'id': self.ids[index],
                'labels': {k: torch.Tensor(v[index].reshape(-1)) for k, v in self.labels.items()}
            }
        else:
            sample = {
                'raw_text': self.rawText[index],
                'text': torch.Tensor(self.text[index]),
                'audio': torch.Tensor(self.audio[index]),
                'vision': torch.Tensor(self.vision[index]),
                'index': index,
                'id': self.ids[index],
                'labels': {k: torch.Tensor(v[index].reshape(-1)) for k, v in self.labels.items()}
            }
            if not self.args.need_data_aligned:
                sample['audio_lengths'] = self.audio_lengths[index]
                sample['vision_lengths'] = self.vision_lengths[index]
        return sample


def MMDataLoader(args):
    datasets = {
        'train': MMDataset(args, mode='train'),
        'valid': MMDataset(args, mode='valid'),
        'test': MMDataset(args, mode='test')
    }

    if 'seq_lens' in args:
        args.seq_lens = datasets['train'].get_seq_len()

    dataLoader = {
        ds: DataLoader(datasets[ds],
                       batch_size=args.batch_size,
                       num_workers=args.num_workers,
                       shuffle=True,
                       persistent_workers=False)
        for ds in datasets.keys()
    }

    return dataLoader