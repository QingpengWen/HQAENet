# -*- coding: utf-8 -*-
"""
@CreateTime :       2025/06/03 16:15
@File       :       AttentionLayer.py
@Software   :       PyCharm
@Framework  :       Pytorch
@LastModify :       2025/07/30 23:35
"""

import torch
from torch import nn
import torch.nn.functional as F

class GroupQueryAttention(nn.Module):

    def __init__(self, embed_dim, num_heads=8, num_groups=4, dropout=0.1):
        super(GroupQueryAttention, self).__init__()
        assert embed_dim % num_heads == 0, "embed_dim must be num_heads"
        assert num_heads % num_groups == 0, "num_heads must be num_groups"

        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.num_groups = num_groups
        self.head_dim = embed_dim // num_heads
        self.group_size = num_heads // num_groups
        self.scale_factor = 1.0 / (self.head_dim ** 0.5)

        # project
        self.q_proj = nn.Linear(embed_dim, embed_dim)
        self.kv_proj = nn.Linear(embed_dim, 2 * embed_dim)
        self.out_proj = nn.Linear(embed_dim, embed_dim)

        # Dropout
        self.dropout = nn.Dropout(dropout)

    def forward(self, query, key, value, key_padding_mask=None):
        """
        Input:
            query: [batch_size, tgt_len, embed_dim]
            key: [batch_size, src_len, embed_dim]
            value: [batch_size, src_len, embed_dim]
        Output:
            attn_output: [batch_size, tgt_len, embed_dim]
        """
        batch_size, tgt_len, _ = query.shape
        src_len = key.size(1)

        q = self.q_proj(query)  # [batch, tgt_len, embed_dim]
        # [batch, groups, tgt_len, group_size, head_dim]
        q = q.view(batch_size, tgt_len, self.num_groups, self.group_size, self.head_dim)
        q = q.permute(0, 2, 1, 3, 4)  # [batch, groups, tgt_len, group_size, head_dim]
        kv = self.kv_proj(key)  # [batch, src_len, 2 * embed_dim]
        k, v = kv.chunk(2, dim=-1)  # [batch, src_len, embed_dim]
        k = k.view(batch_size, src_len, self.num_groups, self.group_size, self.head_dim)
        k = k.permute(0, 2, 3, 4, 1)  # [batch, groups, group_size, head_dim, src_len]
        # [batch, groups, group_size, src_len, head_dim]
        v = v.view(batch_size, src_len, self.num_groups, self.group_size, self.head_dim)
        v = v.permute(0, 2, 3, 1, 4)  # [batch, groups, group_size, src_len, head_dim]

        # [batch, groups, group_size, tgt_len, head_dim]
        q = q.permute(0, 1, 3, 2, 4)  # [batch, groups, group_size, tgt_len, head_dim]
        # [batch, groups, group_size, tgt_len, src_len]
        attn_scores = torch.matmul(
            q,  # [batch, groups, group_size, tgt_len, head_dim]
            k  # [batch, groups, group_size, head_dim, src_len]
        )  # -> [batch, groups, group_size, tgt_len, src_len]
        attn_scores = attn_scores * self.scale_factor

        # mask
        if key_padding_mask is not None:
            mask = key_padding_mask.unsqueeze(1).unsqueeze(1).unsqueeze(1)  # [batch, 1, 1, 1, src_len]
            mask = mask.expand(-1, self.num_groups, self.group_size, tgt_len, -1)
            attn_scores = attn_scores.masked_fill(mask, float('-inf'))

        attn_weights = F.softmax(attn_scores, dim=-1)
        attn_weights = self.dropout(attn_weights)

        # [batch, groups, group_size, tgt_len, head_dim]
        attn_output = torch.matmul(attn_weights, v)  # [batch, groups, group_size, tgt_len, head_dim]

        # [batch, tgt_len, num_heads, head_dim]
        attn_output = attn_output.permute(0, 3, 1, 2, 4)  # [batch, tgt_len, groups, group_size, head_dim]
        attn_output = attn_output.contiguous().view(
            batch_size, tgt_len, self.num_heads * self.head_dim
        )  # [batch, tgt_len, embed_dim]

        attn_output = self.out_proj(attn_output)
        return attn_output

class SelfAttentionEncoder(nn.Module):

    def __init__(self, input_dim, hidden_dim, num_heads=8, num_groups=4, dropout=0.1):
        super(SelfAttentionEncoder, self).__init__()
        self.norm1 = nn.LayerNorm(input_dim)
        self.attn = GroupQueryAttention(
            embed_dim=input_dim,
            num_heads=num_heads,
            num_groups=num_groups,
            dropout=dropout
        )
        self.norm2 = nn.LayerNorm(input_dim)
        self.ffn = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, input_dim),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        residual = x
        x = self.norm1(x)
        attn_output = self.attn(x, x, x)
        x = residual + attn_output
        residual = x
        x = self.norm2(x)
        ffn_output = self.ffn(x)
        x = residual + ffn_output

        return x


class NewSelfAttentionEncoder(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_heads=8, dropout=0.1):
        super(NewSelfAttentionEncoder, self).__init__()
        self.norm1 = nn.LayerNorm(input_dim)
        self.attn = nn.MultiheadAttention(
            embed_dim=input_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True
        )

        self.norm2 = nn.LayerNorm(input_dim)
        self.ffn = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, input_dim),
            nn.Dropout(dropout)
        )

    def forward(self, x, return_attn=False):
        """
        Input:
            x: [batch, seq_len, dim]
            return_attn: whether return attention weights
        Output:
            x: [batch, seq_len, dim]
            attn_weights: weight.shape[batch, num_heads, tgt_len, src_len] (if return_attn=True)
        """
        residual = x
        x_norm = self.norm1(x)

        attn_output, attn_weights = self.attn(
            x_norm, x_norm, x_norm,
            need_weights=return_attn
        )

        x = residual + attn_output
        residual = x
        x = self.norm2(x)
        ffn_output = self.ffn(x)
        x = residual + ffn_output

        if return_attn:
            return x, attn_weights

        return x

class GQAFusion(nn.Module):
    def __init__(self, text_dim=768, visual_dim=32, acoustic_dim=16,
                 hidden_dim=256, num_heads=8, num_groups=4, dropout=0.1):
        super(GQAFusion, self).__init__()
        self.text_encoder = SelfAttentionEncoder(
            text_dim, hidden_dim, num_heads, num_groups, dropout
        )
        self.visual_encoder = SelfAttentionEncoder(
            visual_dim, hidden_dim, num_heads, num_groups, dropout
        )
        self.acoustic_encoder = SelfAttentionEncoder(
            acoustic_dim, hidden_dim, num_heads, num_groups, dropout
        )
        self.text_proj = nn.Linear(text_dim, hidden_dim)
        self.visual_proj = nn.Linear(visual_dim, hidden_dim)
        self.acoustic_proj = nn.Linear(acoustic_dim, hidden_dim)
        self.gqa_text = GroupQueryAttention(
            hidden_dim, num_heads, num_groups, dropout
        )
        self.gqa_visual = GroupQueryAttention(
            hidden_dim, num_heads, num_groups, dropout
        )
        self.gqa_acoustic = GroupQueryAttention(
            hidden_dim, num_heads, num_groups, dropout
        )
        self.text_re = nn.Linear(hidden_dim, text_dim, )
        self.visual_re = nn.Linear(hidden_dim, visual_dim, )
        self.acoustic_re = nn.Linear(hidden_dim, acoustic_dim, )
        self.fusion = nn.Sequential(
            nn.Linear(3 * hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout)
        )

    def forward(self, T, V, A):
        """
        Input:
            T: [batch, seq_len, text_dim]
            V: [batch, seq_len, visual_dim]
            A: [batch, seq_len, acoustic_dim]
        Output:
            fused: [batch, hidden_dim]
        """
        T_proj = self.text_proj(T)
        V_proj = self.visual_proj(V)
        A_proj = self.acoustic_proj(A)

        T_aligned = self.gqa_text(
            T_proj,
            V_proj,
            A_proj
        )

        V_aligned = self.gqa_visual(
            V_proj,
            T_proj,
            A_proj
        )

        A_aligned = self.gqa_acoustic(
            A_proj,
            T_proj,
            V_proj
        )

        T_pool = torch.mean(T_aligned, dim=1)
        V_pool = torch.mean(V_aligned, dim=1)
        A_pool = torch.mean(A_aligned, dim=1)

        fused = torch.cat([T_pool, V_pool, A_pool], dim=1)
        fused = self.fusion(fused)

        return fused, self.text_re(T_aligned), self.visual_re(V_aligned), self.acoustic_re(A_aligned)
