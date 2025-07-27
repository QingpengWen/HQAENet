# -*- coding: utf-8 -*-
"""
@CreateTime :       2025/06/08 21:35
@File       :       GQALayer.py
@Software   :       PyCharm
@Framework  :       Pytorch
@LastModify :       2025/07/30 23:35
"""

import os
import sys
import collections
import time
import torch
import torch.nn as nn
from torch.nn.utils.rnn import pad_sequence, pack_padded_sequence, pad_packed_sequence

from models.missingTask.AttentionLayer import SelfAttentionEncoder, GQAFusion
from models.missingTask.HQAENet import HQAENet
from models.missingTask.restoration import VideoShow
from models.subNets.BertTextEncoder import BertTextEncoder
from models.subNets.hgatt import HGAtt
import torch.nn.functional as F

__all__ = ['HQAENetwork']

class HQAENetwork(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.aligned = args.need_data_aligned
        self.model_name = args.modelName
        # Pre-encoders
        ## text Pre-encoder
        self.text_model = BertTextEncoder(language=args.language, use_finetune=args.use_finetune)
        self.layer = args.num_layer
        self.audio_features = 16
        self.visual_features = 32
        self.text_features = 768
        self.attn_dropout = args.attn_dropout
        self.device = args.device
        self.embed_dim = self.visual_features
        self.hidden_dim = 256
        self.num_heads = args.num_heads
        self.num_groups = args.num_groups
        ## audio-visual Pre-encoders
        audio_in, video_in = args.feature_dims[1:]
        self.audio_model = AuViSubNet(audio_in, args.a_lstm_hidden_size, args.audio_out, \
                            num_layers=args.a_lstm_layers, dropout=args.a_lstm_dropout)
        self.video_model = AuViSubNet(video_in, args.v_lstm_hidden_size, args.video_out, \
                            num_layers=args.v_lstm_layers, dropout=args.v_lstm_dropout)

        # HQAENet
        self.HQAELayer = HQAENet(noise_std=0.1)

        # Self-Attention Encoder
        self.text_encoder = SelfAttentionEncoder(
            self.text_features, self.hidden_dim, self.num_heads, self.num_groups, self.attn_dropout
        )
        self.visual_encoder = SelfAttentionEncoder(
            self.visual_features, self.hidden_dim, self.num_heads, self.num_groups, self.attn_dropout
        )
        self.acoustic_encoder = SelfAttentionEncoder(
            self.audio_features, self.hidden_dim, self.num_heads, self.num_groups, self.attn_dropout
        )

        # GQA Reconstructor
        self.gqa_fusion = GQAFusion(
            text_dim=self.text_features,
            visual_dim=self.visual_features,
            acoustic_dim=self.audio_features,
            hidden_dim=self.hidden_dim,
            num_heads=self.num_heads,
            num_groups=self.num_groups
        )

        self.HGAtt = HGAtt(self.audio_features, self.visual_features, self.text_features, self.attn_dropout)
        self.H_linears = nn.Linear(in_features=self.text_features, out_features=self.embed_dim, bias=None)
        self.v_HGAtt = HGAtt(self.text_features, self.audio_features, self.visual_features, self.attn_dropout)
        self.t_HGAtt = HGAtt(self.visual_features, self.audio_features, self.text_features, self.attn_dropout)
        self.a_HGAtt = HGAtt(self.text_features, self.visual_features, self.audio_features, self.attn_dropout)
        self.vt_HGAtt = HGAtt(self.visual_features, self.text_features, self.text_features, self.attn_dropout)
        self.at_HGAtt = HGAtt(self.audio_features, self.text_features, self.text_features, self.attn_dropout)
        self.vta_HGAtt = HGAtt(self.text_features, self.embed_dim, self.embed_dim, self.attn_dropout)

        self.layer_norm = nn.LayerNorm(self.embed_dim)
        self.a_linear = nn.Linear(in_features=self.audio_features, out_features=self.embed_dim, bias=None)
        self.v_linear = nn.Linear(in_features=self.visual_features, out_features=self.embed_dim, bias=None)
        self.t_linear = nn.Linear(in_features=self.text_features, out_features=self.embed_dim, bias=None)
        if self.model_name == "hgatt_net":
            self.text_linear = nn.Linear(in_features=self.embed_dim, out_features=self.text_features, bias=None)
            self.audio_linear = nn.Linear(in_features=self.embed_dim, out_features=self.audio_features, bias=None)
        self.va_linear = nn.Linear(in_features=self.embed_dim, out_features=self.embed_dim, bias=None)  # True
        self.h_linear = nn.Linear(in_features=self.embed_dim, out_features=self.embed_dim, bias=None)  # True
        self.ht_linear = nn.Linear(in_features=self.embed_dim, out_features=self.embed_dim, bias=None)
        self.hv_linear = nn.Linear(in_features=self.embed_dim, out_features=self.embed_dim, bias=None)
        self.ha_linear = nn.Linear(in_features=self.embed_dim, out_features=self.embed_dim, bias=None)  # True
        self.Ht_linear = nn.Linear(in_features=self.text_features, out_features=self.embed_dim, bias=None)
        self.Hv_linear = nn.Linear(in_features=self.visual_features, out_features=self.embed_dim, bias=None)
        self.Ha_linear = nn.Linear(in_features=self.audio_features, out_features=self.embed_dim, bias=None)  # True
        self.multimodal_classifier = nn.Sequential(nn.Linear(self.embed_dim, 1))
        self.text_classifier = nn.Sequential(nn.Linear(self.embed_dim, self.text_features))
        self.audio_classifier = nn.Sequential(nn.Linear(self.embed_dim, self.audio_features))
        self.video_classifier = nn.Sequential(nn.Linear(self.embed_dim, self.visual_features))
        self.fine_classifier = nn.Sequential(nn.Linear(self.embed_dim, self.embed_dim))
        self.CrossAttention = CrossAttention(input_dim_a=self.embed_dim, input_dim_b=self.embed_dim, hidden_dim=self.embed_dim)
        self.recon_text = nn.Linear(self.embed_dim, args.feature_dims[0])
        self.recon_audio = nn.Linear(self.embed_dim, args.feature_dims[1])
        self.recon_video = nn.Linear(self.embed_dim, args.feature_dims[2])

        self.W = nn.Sequential()
        self.W.add_module('discriminator_layer_2',
                          nn.Linear(self.embed_dim, out_features=1, bias=False))

        self.DiscriminatorHT = AdversarialDiscriminator(self.text_features, self.embed_dim)
        self.DiscriminatorHV = AdversarialDiscriminator(self.visual_features, self.embed_dim)
        self.DiscriminatorHA = AdversarialDiscriminator(self.audio_features, self.embed_dim)
        self.DiscriminatorET = AdversarialDiscriminator(self.text_features, self.embed_dim)
        self.DiscriminatorEV = AdversarialDiscriminator(self.visual_features, self.embed_dim)
        self.DiscriminatorEA = AdversarialDiscriminator(self.audio_features, self.embed_dim)

        self.VideoShow = VideoShow(input_dim=self.text_features, batch_size=args.batch_size)

    def forward_once(self, text, text_lengths, audio, audio_lengths, video, video_lengths, missing, do_test=None):

        # TODO: Pre-Encoder
        text = self.text_model(text)   # [batch, 50, 768]
        text_utt = text[:, 0]  # (B, 1, D), (B, T, D)
        max_length = 50
        batch_number = audio_lengths.numel()
        lengths = torch.linspace(3, max_length, steps=batch_number)
        audio_lengths = video_lengths = lengths.to(self.device)
        if int(text.shape[1]) < max_length:
            padding_length = max_length - text.shape[1]
            padding = torch.zeros(text.shape[0], padding_length, text.shape[2], dtype=text.dtype, device=text.device)
            text = torch.cat((text, padding), dim=1)
        text_for_recon = text.detach()

        audio, audio_utt = self.audio_model(audio, audio_lengths,
                                            return_temporal=True)  # audio[batch, 68, 16] utt[batch, 16]-->unaligned
        video, video_utt = self.video_model(video, video_lengths, return_temporal=True)  # [batch, 81, 32]

        # TODO: input C_{V,T,A}
        input_audio = audio  # [batch, 50, 16]
        input_visual = video  # [batch, 50, 32]
        input_text = text  # [batch, 50, 768]

        ## TODO: HQAENet
        H_text, H_visual, H_audio, final_visual_sim, final_acoustic_sim = self.HQAELayer(input_text, input_visual, input_audio)

        # TODO: Encoder
        H_text = self.text_encoder(H_text)  # [batch, 50, 768]
        H_visual = self.visual_encoder(H_visual)  # [batch, 50, 32]
        H_audio = self.acoustic_encoder(H_audio)  # [batch, 50, 16]

        # TODO: Regular Reconstruction-based Model
        if self.model_name == "gqa_net":
            """"
            Example:
            Group Query Attention (GQA)
            """
            fused, H_text, H_visual, H_audio, = self.gqa_fusion(H_text, H_visual, H_audio)
        elif self.model_name == "hgatt_net":
            """
            Example:
            Higher-order Gate Attention (HGAtt)
            Reproduced from the GitHub project, not represent the official code.
            """
            H_visual_text = self.vt_HGAtt(input_visual, input_text, input_text, layer=self.layer)[0]  # [batch, 50, 768]
            H_visual_text = self.t_linear(H_visual_text)  # [batch, 50, 32]
            H_visual = H_visual_text
            H_audio_text = self.at_HGAtt(input_audio, input_text, input_text, layer=self.layer)[0]  # [batch, 50, 74]
            H_audio_text = self.Ht_linear(H_audio_text)  # [batch, 50, 32]
            H_audio = self.audio_linear(H_audio_text)
            H_visual_audio = self.CrossAttention(H_visual_text, H_visual_text)
            H_out = self.vta_HGAtt(input_text, H_visual_audio[0], H_visual_audio[1], layer=int(self.layer + 5))[1]  # [batch, 50, 32]
            H_text = self.text_linear(H_out + self.va_linear(H_visual_audio[2]))  # [batch, 50, 32]
        else:
            raise ValueError("Please enter reconstruction models")

        # TODO: Discriminator
        self.D_ht = F.softmax(self.DiscriminatorHT(H_text), dim=1)
        H_text = self.t_linear(H_text)  # [batch, 50, 32]
        self.D_hv = F.softmax(self.DiscriminatorHV(H_visual), dim=1)
        H_visual = self.v_linear(H_visual)  # [batch, 50, 32]
        self.D_ha = F.softmax(self.DiscriminatorHA(H_audio), dim=1)
        H_audio = self.a_linear(H_audio)  # [batch, 50, 32]

        #  TODO: FFN Classifier
        H = self.ha_linear(H_audio) + self.hv_linear(H_visual) + self.ht_linear(H_text)

        H_text_logit = H_text.permute(1, 0, 2)  # [50, batch, 32]
        H_visual_logit = H_visual.permute(1, 0, 2)  # [50, batch, 32]
        H_audio_logit = H_audio.permute(1, 0, 2)  # [50, batch, 32]
        H_logit = H.permute(1, 0, 2)  # [50, batch, 32]
        H_logit = H_logit[-1]
        H_text_logit = H_text_logit[-1]
        H_visual_logit = H_visual_logit[-1]
        H_audio_logit = H_audio_logit[-1]
        predit_out = self.h_linear(H_logit)

        text_predit_out = self.text_classifier(H_text_logit)
        visual_predit_out = self.video_classifier(H_visual_logit)
        audio_predit_out = self.audio_classifier(H_audio_logit)
        fine_predit_out = self.fine_classifier(H_logit)
        predit_out_hidden = predit_out
        predit_out = self.multimodal_classifier(predit_out)

        output_fusion = predit_out
        z_text = text_utt
        p_text = text_predit_out
        z_audio = audio_utt
        p_audio = audio_predit_out
        z_video = video_utt
        p_video = visual_predit_out
        z_gmc_tokens = H_logit
        p_gmc_tokens = fine_predit_out

        suffix = '_m' if missing else ''
        res = {
            f'pred{suffix}': output_fusion,  # [batch, 1] # 预测输出值
            f'z_gmc_tokens{suffix}': z_gmc_tokens.detach(),  # [batch, 384] # 三个模态融合输出值
            f'p_gmc_tokens{suffix}': p_gmc_tokens,  # [batch, 384] # 三个模态融合预测输出值
            f'z_text{suffix}': z_text.detach(),  # [batch, 768] # 文本原始输出值
            f'p_text{suffix}': p_text,  # [batch, 768] # 文本预测输出值
            f'z_audio{suffix}': z_audio.detach(),  # [batch, 16]
            f'p_audio{suffix}': p_audio,  # [batch, 16]
            f'z_video{suffix}': z_video.detach(),  # [batch, 32]
            f'p_video{suffix}': p_video,  # [batch, 32]
            f'pred_hidden{suffix}': predit_out_hidden,
        }

        # low-level feature reconstruction
        if missing:
            text_recon = self.recon_text(H_text)
            audio_recon = self.recon_audio(H_audio)
            video_recon = self.recon_video(H_visual)
            res.update(
                {
                    'text_recon': text_recon,
                    'audio_recon': audio_recon,
                    'video_recon': video_recon,
                }
            )
        else:
            res.update({'text_for_recon': text_for_recon})

        return res

    def forward(self, text, audio, video, do_test):
        text, text_m = text  # [batch, 3, 50]
        audio, audio_m, audio_lengths = audio  # [batch, 50, 5]
        video, video_m, video_lengths = video  # [batch, 50, 20]

        mask_len = torch.sum(text[:, 1, :], dim=1, keepdim=True)
        text_lengths = mask_len.squeeze().int().detach() - 2 # -2 for CLS and SEP

        # complete view
        res = self.forward_once(text, text_lengths, audio, audio_lengths, video, video_lengths,missing=False, do_test=do_test)
        # incomplete view
        res_m = self.forward_once(text_m, text_lengths, audio_m, audio_lengths, video_m, video_lengths, missing=True, do_test=do_test)

        return {**res, **res_m}

class AdversarialDiscriminator(nn.Module):
    def __init__(self, feature_dim, out_dim):
        super(AdversarialDiscriminator, self).__init__()
        self.fc1 = nn.Linear(feature_dim, 64)
        self.fc2 = nn.Linear(64, 32)
        self.fc3 = nn.Linear(32, out_dim)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.sigmoid(self.fc3(x))

class AuViSubNet(nn.Module):
    def __init__(self, in_size, hidden_size, out_size=None, num_layers=1, dropout=0.2, bidirectional=False):
        '''
        Args:
            in_size: input dimension
            hidden_size: hidden layer dimension
            num_layers: specify the number of layers of LSTMs.
            dropout: dropout probability
            bidirectional: specify usage of bidirectional LSTM
        Output:
            (return value in forward) a tensor of shape (batch_size, out_size)
        '''
        super().__init__()
        self.rnn = nn.LSTM(in_size, hidden_size, num_layers=num_layers, dropout=dropout, bidirectional=bidirectional, batch_first=True)
        self.dropout = nn.Dropout(dropout)
        feature_size = hidden_size * 2 if bidirectional else hidden_size
        self.linear_1 = nn.Linear(feature_size, out_size) if feature_size != out_size and out_size is not None else nn.Identity()

    def forward(self, x, lengths, return_temporal=False):
        '''
        x: (batch_size, sequence_len, in_size)
        '''
        # for pytorch1.2
        # packed_sequence = pack_padded_sequence(x, lengths, batch_first=True, enforce_sorted=False)
        # for pytorch1.7
        # a = lengths
        packed_sequence = pack_padded_sequence(x, lengths.cpu(), batch_first=True, enforce_sorted=False)
        packed_last_hidden_state, final_states = self.rnn(packed_sequence)

        h = self.dropout(final_states[0].squeeze())
        y_1 = self.linear_1(h)
        if not return_temporal:
            return y_1
        else:
            unpacked_last_hidden_state, _ = pad_packed_sequence(packed_last_hidden_state, batch_first=True)
            last_hidden_state = self.linear_1(unpacked_last_hidden_state)
            return last_hidden_state, y_1


class Projector(nn.Module):
    def __init__(self, input_dim, output_dim):
        super().__init__()

        self.net = nn.Sequential(nn.Linear(input_dim, output_dim),
                                 nn.BatchNorm1d(output_dim, affine=False))

    def forward(self, x):
        return self.net(x)


class Predictor(nn.Module):
    def __init__(self, input_dim, pred_dim, output_dim):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(input_dim, pred_dim, bias=False),
                                 nn.BatchNorm1d(pred_dim),
                                 nn.ReLU(inplace=True),  # hidden layer
                                 nn.Linear(pred_dim, output_dim))  # output layer

    def forward(self, x):
        return self.net(x)


class CrossAttention(nn.Module):
    def __init__(self, input_dim_a, input_dim_b, hidden_dim):
        super(CrossAttention, self).__init__()

        self.linear_a = nn.Linear(input_dim_a, hidden_dim)
        self.linear_b = nn.Linear(input_dim_b, hidden_dim)
        self.linear_ab = nn.Linear(hidden_dim, hidden_dim, bias=True)
        self.linear_ba = nn.Linear(hidden_dim, hidden_dim, bias=False)

    def forward(self, input_a, input_b):
        mapped_a = self.linear_a(input_a)  # (batch_size, seq_len_a, hidden_dim)
        mapped_b = self.linear_b(input_b)  # (batch_size, seq_len_b, hidden_dim)
        y = mapped_b.transpose(1, 2)

        scores = torch.matmul(mapped_a, mapped_b.transpose(1, 2))  # (batch_size, seq_len_a, seq_len_b)
        attentions_a = torch.softmax(scores, dim=-1)
        attentions_b = torch.softmax(scores.transpose(1, 2),
                                     dim=-1)
        output_a = torch.matmul(attentions_b, input_b)  # (batch_size, seq_len_a, input_dim_b)
        output_b = torch.matmul(attentions_a.transpose(1, 2), input_a)  # (batch_size, seq_len_b, input_dim_a)

        output_ab = self.linear_ab(output_a) + self.linear_ba(output_b)

        return output_a, output_b, output_ab

def length_to_mask(length, max_len=None, dtype=None):
    """length: B.
    return B x max_len.
    If max_len is None, then max of length will be used.
    """
    assert len(length.shape) == 1, 'Length shape should be 1 dimensional.'
    max_len = max_len or length.max().item()
    mask = torch.arange(max_len, device=length.device,
                        dtype=length.dtype).expand(len(length), max_len) < length.unsqueeze(1)
    if dtype is not None:
        mask = torch.as_tensor(mask, dtype=dtype, device=length.device)
    return mask