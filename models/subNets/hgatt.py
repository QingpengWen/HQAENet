# -*- coding: utf-8 -*-
"""
@CreateTime :       2025/07/06 17:35
@File       :       hgatt.py
@Software   :       PyCharm
@Framework  :       Pytorch
@LastModify :       2025/07/30 23:35
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

def LayerNorm(embedding_dim):
    m = nn.LayerNorm(embedding_dim)
    return m

class SingleGAtt(nn.Module):
    def __init__(self, Q_features, K_features, V_features, scale, dropout_rate):
        super(SingleGAtt, self).__init__()

        # self.scale = scale
        self.scale = np.power(K_features, 0.5)
        self.__dropout_layer = nn.Dropout(dropout_rate)
        
        self.embed_dim = V_features
        self.Q_linear = nn.Linear(in_features=Q_features, out_features=self.embed_dim, bias=None)
        self.K_linear = nn.Linear(in_features=K_features, out_features=self.embed_dim, bias=None)
        self.V_linear = nn.Linear(in_features=V_features, out_features=self.embed_dim, bias=None)
        self.h_linear = nn.Linear(in_features=self.embed_dim + self.embed_dim, out_features=self.embed_dim, bias=True)
        self.v_linear = nn.Linear(in_features=self.embed_dim, out_features=self.embed_dim, bias=None)
        self.u_linear = nn.Linear(in_features=50, out_features=50, bias=True)
        self.f_linear = nn.Linear(in_features=50, out_features=self.embed_dim, bias=None)
        self.vq_linear = nn.Linear(in_features=50, out_features=self.embed_dim, bias=None)
        self.vk_linear = nn.Linear(in_features=50, out_features=self.embed_dim, bias=None)
        self.qv_linear = nn.Linear(in_features=50, out_features=Q_features, bias=None)
        self.kv_linear = nn.Linear(in_features=50, out_features=K_features, bias=None)

    def forward(self, Q, K, V, mode, dropout=None):
              
        if mode == 'mode1':
            q = torch.tanh(Q)
            k = torch.tanh(K)
            w_q = self.Q_linear(q)
            w_k = self.K_linear(k)
            h = torch.cat((w_q, w_k), dim=-1)  # [batch, 50, 64]
            fusion_rates = self.h_linear(h)
            f = torch.sigmoid(fusion_rates)
            V_Q = torch.cat((Q, V), dim=-1)
            V_K = torch.cat((K, V), dim=-1)
            V_Q = self.V_linear(V_Q)
            V_K = self.V_linear(V_K)
            q = f * q + V_Q
            k = (1 - f) * k + V_K
            f = torch.sigmoid(self.h_linear(torch.cat((q, k), dim=-1)))
            if dropout:
                q = self.__dropout_layer(q)
                k = self.__dropout_layer(k)
            return q, k, f
        elif mode == 'mode2':
            q_input = torch.tanh(Q)
            k_input = torch.tanh(K)
            v_input = torch.tanh(V)
            q = self.Q_linear(q_input)
            k = self.K_linear(k_input)
            v = self.v_linear(v_input)
            
            qk = torch.bmm(q, k.transpose(1, 2)) / self.scale  
            vq = torch.bmm(q, v.transpose(1, 2)) / self.scale
            vk = torch.bmm(k, v.transpose(1, 2)) / self.scale

            f = F.softmax(qk, dim=-1)
            q = torch.bmm(f, q) + self.vq_linear(vq)
            k = torch.bmm((1 - f), k) + self.vk_linear(vk)
            
            Q_predit_out = torch.bmm(f, q)
            K_predit_out = torch.bmm((1 - f), k)

            if dropout:
                Q_predit_out = self.__dropout_layer(Q_predit_out)
                K_predit_out = self.__dropout_layer(K_predit_out)
            return Q_predit_out, K_predit_out, f
            
    
class HGAtt(nn.Module):
    def __init__(self, Q_features, K_features, V_features, dropout_rate=0.1):
        super(HGAtt, self).__init__()
        
        self.dropout_rate = dropout_rate
        self.scale = np.power(K_features, 0.5)
        self.embed_dim = V_features
        self.gatt = SingleGAtt(Q_features, K_features, V_features, self.scale, self.dropout_rate)

        self.Q_linear = nn.Linear(in_features=Q_features, out_features=self.embed_dim, bias=None)
        self.K_linear = nn.Linear(in_features=K_features, out_features=self.embed_dim, bias=None)
        self.V_linear = nn.Linear(in_features=V_features, out_features=self.embed_dim, bias=None)

        self.layer_norm = nn.LayerNorm(self.embed_dim)
        self.layer_norms = nn.ModuleList([LayerNorm(self.embed_dim) for _ in range(2)])
        self.fc1 = nn.Linear(self.embed_dim, 4*self.embed_dim)   # The "Add & Norm" part in the paper
        self.fc2 = nn.Linear(4*self.embed_dim, self.embed_dim)
        
        self.normalize_before = True

        self.multimodal_classifier = nn.Sequential(
            nn.Linear(self.embed_dim, 1)
        )

    def maybe_layer_norm(self, i, x, before=False, after=False):
        assert before ^ after
        if after ^ self.normalize_before:
            return self.layer_norms[i](x)
        else:
            return x
    
    def forward(self, Q, K, V, layer):
        mode = 'mode2'
        Q_input = self.Q_linear(Q)  # [batch, 50, 32]
        K_input = self.K_linear(K)  # [batch, 50, 32]
        V_input = self.V_linear(V)

        residual = V_input
        
        for i in range(layer): 
            q, k, f = self.gatt(Q, K, V, mode)
            Q_output, K_output = q, k

        if mode == 'mode1':
            H = f * q + (1 - f) * k
            H = self.layer_norm(H)  # [batch, 50, 32]
        elif mode == 'mode2':

            u = torch.bmm(Q_output, K_output.transpose(1, 2))
            u = u / self.scale
            att = F.softmax(u, dim=-1)
            H_hgatt = torch.bmm(att, V_input)
            
            # Add & Normal
            H = F.dropout(H_hgatt, p=self.dropout_rate, training=self.training)
            H = residual + H
            H = self.maybe_layer_norm(0, H, after=True)

            residual = H
            H = self.maybe_layer_norm(1, H, before=True)
            H = F.relu(self.fc1(H))
            H = F.dropout(H, p=self.dropout_rate, training=self.training)
            H = self.fc2(H)
            H = F.dropout(H, p=self.dropout_rate, training=self.training)
            H = residual + H
            H = self.maybe_layer_norm(1, H, after=True)
            H = self.layer_norm(H)
        
        return H, H_hgatt
