# -*- coding: utf-8 -*-
"""
Created on Dec 14 2020 
@author: Yi-Hui (Sophia) Chou

Updated on May 10 2021
@author: I-Chun (Bronwin) Chen
"""
import math
import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F

from torch.nn import init
from torch.nn import LSTM, Linear  
from ops import ConvolutionLayer, MaxOverTimePooling, Conv1d_mp, BiLSTM, ConvBlock, SelfAttention
from torch.nn.utils.rnn import pad_packed_sequence, pack_padded_sequence, PackedSequence


class Embeddings(nn.Module):
    def __init__(self, n_token, d_model):
        super(Embeddings, self).__init__()
        self.lut = nn.Embedding(n_token, d_model)
        self.d_model = d_model

    def forward(self, x):
        return self.lut(x) * math.sqrt(self.d_model)

class SAN(nn.Module):
    def __init__(self, num_of_dim, e2w, vocab_size, embedding_size, r, lstm_hidden_dim=128, da=128, hidden_dim=256) -> None:
        super(SAN, self).__init__()

        self.e2w = e2w
        self.n_tokens = []
        for key in e2w:
            self.n_tokens.append(len(e2w[key]))
        self.emb_sizes = [256, 256, 256, 256]

        self.word_emb = []
        for i, key in enumerate(e2w):
            self.word_emb.append(Embeddings(self.n_tokens[i], self.emb_sizes[i]))
        self.word_emb = nn.ModuleList(self.word_emb)

        self.in_linear = nn.Linear(np.sum(self.emb_sizes), embedding_size)

        self.proj = []
        for i, etype in enumerate(e2w):
            self.proj.append(nn.Linear(embedding_size, self.n_tokens[i]))
        self.proj = nn.ModuleList(self.proj)

        self._bilstm = nn.LSTM(embedding_size, lstm_hidden_dim, batch_first=True, bidirectional=True)
        self._attention = SelfAttention(2 * lstm_hidden_dim, da, r)
        self._classifier = nn.Sequential(
            nn.Linear(2 * lstm_hidden_dim * r, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, num_of_dim)
        )
        
    def forward(self, x: torch.Tensor):
        embs = []
        for i, key in enumerate(self.e2w):
            embs.append(self.word_emb[i](x[..., i]))
        embs = torch.cat([*embs], dim=-1)
        emb_linear = self.in_linear(embs)

        outputs, hc = self._bilstm(emb_linear)
        attn_mat = self._attention(outputs)
        m = torch.bmm(attn_mat, outputs)
        flatten = m.view(m.size()[0], -1)
        score = self._classifier(flatten)
        return score

    def _get_attention_weight(self, x):
        embs = []
        for i, key in enumerate(self.e2w):
            embs.append(self.word_emb[i](x[..., i]))
        embs = torch.cat([*embs], dim=-1)
        emb_linear = self.in_linear(embs)

        outputs, hc = self._bilstm(emb_linear)
        attn_mat = self._attention(outputs)
        m = torch.bmm(attn_mat, outputs)
        flatten = m.view(m.size()[0], -1)
        score = self._classifier(flatten)
        return score, attn_mat
