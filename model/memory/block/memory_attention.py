# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import math
import copy

import torch
import torch.nn as nn
import torch.nn.functional as F


def get_activation_fn(activation):
    """Return an activation function given a string"""
    if activation == "relu":
        return F.relu
    if activation == "gelu":
        return F.gelu
    if activation == "glu":
        return F.glu
    raise RuntimeError(f"activation should be relu/gelu, not {activation}.")

def get_clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for i in range(N)])


class PositionalEncoding(nn.Module):
    def __init__(self, d_model:int, dropout=0, max_len:int=5000) -> None:

        super(PositionalEncoding, self).__init__()
        
        self.dropout = nn.Dropout(p=dropout)
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1) 
        div_term = torch.exp(torch.arange(0, d_model, 2) * -(math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term) 
        pe[:, 1::2] = torch.cos(position * div_term) 
        pe = pe.unsqueeze(0)  # size=(1, L, d_model)
        self.register_buffer('pe', pe)  

    def forward(self, x):
        #  output = word_embedding + positional_embedding
        x = x + nn.Parameter(self.pe[:, :x.size(1)],requires_grad=False) #size = [batch, L, d_model]
        return self.dropout(x) # size = [batch, L, d_model]


class MemoryAttentionLayer(nn.Module):

    def __init__(
        self,
        activation: str = "relu",
        d_model: int = 256,
        dim_feedforward: int = 512,
        kv_dim: int = 64,
    ):
        super().__init__()
        self.d_model = d_model
        self.dim_feedforward = dim_feedforward
        self.self_attn = nn.MultiheadAttention(embed_dim=d_model, num_heads=1, batch_first=True)
        self.self_attn_pos = PositionalEncoding(d_model=d_model)

        self.cross_attn_image = nn.MultiheadAttention(embed_dim=d_model, num_heads=4, kdim=kv_dim, vdim=kv_dim, batch_first=True)
        self.cross_attn_image_pos = PositionalEncoding(d_model=kv_dim)

        # Implementation of Feedforward model
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)

        self.activation_str = activation
        self.activation = get_activation_fn(activation)
        self.scale = nn.Parameter(1.421 * torch.ones((d_model)), requires_grad=True)

    def _forward_sa(self, tgt):
        # Self-Attention
        tgt2 = self.norm1(tgt)
        q = k = self.self_attn_pos(tgt2)
        tgt2 = self.self_attn(
            query=q, 
            key=k, 
            value=tgt2
        )
        return tgt

    def _forward_ca(self, tgt, memory):
        # Cross-Attention
        tgt2 = self.norm2(tgt)
        tgt2 = self.cross_attn_image(
            query=tgt2,
            key=self.cross_attn_image_pos(memory),
            value=memory,
        )
        return tgt

    def forward(
        self,
        tgt,
        memory,
        num_k_exclude_rope: int = 0,
    ) -> torch.Tensor:

        # Self-Attn, Cross-Attn
        tgt = self._forward_sa(tgt)
        ca_tgt = self._forward_ca(tgt, memory)
        # learnable scale
        tgt = tgt + self.scale * ca_tgt
        # print(self.scale)

        # MLP
        tgt2 = self.norm3(tgt)
        tgt2 = self.linear2(self.activation(self.linear1(tgt2)))
        return tgt


class MemoryAttention(nn.Module):
    def __init__(
        self,
        d_model: int,
        layer: nn.Module,
        num_layers: int,
    ):
        super().__init__()
        self.d_model = d_model
        self.layers = get_clones(layer, num_layers)
        self.num_layers = num_layers
        self.norm = nn.LayerNorm(d_model)

    def forward(
        self,
        curr: torch.Tensor,  # self-attention inputs
        memory: torch.Tensor,  # cross-attention inputs
    ):
        output = curr

        for layer in self.layers:
            output = layer(
                tgt=output,
                memory=memory,
            )
        normed_output = self.norm(output)

        return normed_output
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import math
import copy

import torch
import torch.nn as nn
import torch.nn.functional as F


def get_activation_fn(activation):
    """Return an activation function given a string"""
    if activation == "relu":
        return F.relu
    if activation == "gelu":
        return F.gelu
    if activation == "glu":
        return F.glu
    raise RuntimeError(f"activation should be relu/gelu, not {activation}.")

def get_clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for i in range(N)])


class PositionalEncoding(nn.Module):
    def __init__(self, d_model:int, dropout=0, max_len:int=5000) -> None:

        super(PositionalEncoding, self).__init__()
        
        self.dropout = nn.Dropout(p=dropout)
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1) 
        div_term = torch.exp(torch.arange(0, d_model, 2) * -(math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term) 
        pe[:, 1::2] = torch.cos(position * div_term) 
        pe = pe.unsqueeze(0)  # size=(1, L, d_model)
        self.register_buffer('pe', pe)  

    def forward(self, x):
        #  output = word_embedding + positional_embedding
        x = x + nn.Parameter(self.pe[:, :x.size(1)],requires_grad=False) #size = [batch, L, d_model]
        return self.dropout(x) # size = [batch, L, d_model]


class MemoryAttentionLayer(nn.Module):

    def __init__(
        self,
        activation: str = "relu",
        d_model: int = 256,
        dim_feedforward: int = 512,
        kv_dim: int = 64,
    ):
        super().__init__()
        self.d_model = d_model
        self.dim_feedforward = dim_feedforward
        self.self_attn = nn.MultiheadAttention(embed_dim=d_model, num_heads=1, batch_first=True)
        self.self_attn_pos = PositionalEncoding(d_model=d_model)

        self.cross_attn_image = nn.MultiheadAttention(embed_dim=d_model, num_heads=4, kdim=kv_dim, vdim=kv_dim, batch_first=True)
        self.cross_attn_image_pos = PositionalEncoding(d_model=kv_dim)

        # Implementation of Feedforward model
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)

        self.activation_str = activation
        self.activation = get_activation_fn(activation)
        self.scale = nn.Parameter(1.421 * torch.ones((d_model)))

    def _forward_sa(self, tgt):
        # Self-Attention
        tgt2 = self.norm1(tgt)
        q = k = self.self_attn_pos(tgt2)
        tgt2 = self.self_attn(
            query=q, 
            key=k, 
            value=tgt2
        )
        return tgt

    def _forward_ca(self, tgt, memory):
        # Cross-Attention
        tgt2 = self.norm2(tgt)
        tgt2 = self.cross_attn_image(
            query=tgt2,
            key=self.cross_attn_image_pos(memory),
            value=memory,
        )
        return tgt

    def forward(
        self,
        tgt,
        memory,
        num_k_exclude_rope: int = 0,
    ) -> torch.Tensor:

        # Self-Attn, Cross-Attn
        tgt = self._forward_sa(tgt)
        ca_tgt = self._forward_ca(tgt, memory)
        # learnable scale
        tgt = tgt + self.scale * ca_tgt
        # print(self.scale)

        # MLP
        tgt2 = self.norm3(tgt)
        tgt2 = self.linear2(self.activation(self.linear1(tgt2)))
        return tgt


class MemoryAttention(nn.Module):
    def __init__(
        self,
        d_model: int,
        layer: nn.Module,
        num_layers: int,
    ):
        super().__init__()
        self.d_model = d_model
        self.layers = get_clones(layer, num_layers)
        self.num_layers = num_layers
        self.norm = nn.LayerNorm(d_model)

    def forward(
        self,
        curr: torch.Tensor,  # self-attention inputs
        memory: torch.Tensor,  # cross-attention inputs
    ):
        output = curr

        for layer in self.layers:
            output = layer(
                tgt=output,
                memory=memory,
            )
        normed_output = self.norm(output)

        return normed_output
