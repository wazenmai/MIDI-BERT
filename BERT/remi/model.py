import math
import numpy as np
import random

import torch
import torch.nn as nn
from transformers import BertModel

class Embeddings(nn.Module):
    def __init__(self, n_token, d_model):
        super().__init__()
        self.lut = nn.Embedding(n_token, d_model)
        self.d_model = d_model

    def forward(self, x):
        return self.lut(x) * math.sqrt(self.d_model)

# BERT model: similar approach to "felix"
class MidiBert(nn.Module):
    def __init__(self, bertConfig, e2w, w2e):
        super().__init__()
        
        self.bert = BertModel(bertConfig)
        bertConfig.d_model = bertConfig.hidden_size
        self.hidden_size = bertConfig.hidden_size
        self.bertConfig = bertConfig

        # token types: [Tempo, Bar, Position, Pitch, Duration, Velocity]
        self.n_token = len(e2w)
        self.emb_size = 256
        self.e2w = e2w
        self.w2e = w2e

        # for deciding whether the current input_ids is a <PAD> token
        self.pad_word = self.e2w['Pad_None']        
        self.mask_word = self.e2w['Mask_None']
        
        # word_emb: embeddings to change token ids into embeddings
        self.word_emb = Embeddings(self.n_token, self.emb_size) 

        # linear layer to merge embeddings from different token types to feed into transformer-XL
        self.in_linear = nn.Linear(self.emb_size, bertConfig.d_model)


    def forward(self, input_id, attn_mask=None, output_hidden_states=True):
        # convert input_ids into embeddings and merge them through linear layer
        emb = self.word_emb(input_id)
        emb_linear = self.in_linear(emb)
        
        # feed to bert 
        y = self.bert(inputs_embeds=emb_linear, attention_mask=attn_mask, output_hidden_states=output_hidden_states)
        #y = y.last_hidden_state         # (batch_size, seq_len, 768)

        return y
