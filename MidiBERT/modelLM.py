import math
import numpy as np
import random

import torch
import torch.nn as nn
from transformers import BertModel

from MidiBERT.model import MidiBert_CP, MidiBert_remi


class MidiBertLM(nn.Module):
    def __init__(self, midibert, rep):
        super().__init__()
        
        self.midibert = midibert
        if rep == 'CP':
            self.mask_lm = MLM_CP(self.midibert.e2w, self.midibert.n_tokens, self.midibert.hidden_size)
        elif rep == 'remi':
            self.mask_lm = MLM_remi(self.midibert.e2w, self.midibert.emb_size, self.midibert.hidden_size)
        else:
            raise NotImplementedError()

    def forward(self, x, attn):
        x = self.midibert(x, attn)
        return self.mask_lm(x)
    

class MLM_CP(nn.Module):
    def __init__(self, e2w, n_tokens, hidden_size):
        super().__init__()
        
        # proj: project embeddings to logits for prediction
        self.proj = []
        for i, etype in enumerate(e2w):
            self.proj.append(nn.Linear(hidden_size, n_tokens[i]))
        self.proj = nn.ModuleList(self.proj)

        self.e2w = e2w
    
    def forward(self, y):
        # feed to bert 
        y = y.hidden_states[-1]
        
        # convert embeddings back to logits for prediction
        ys = []
        for i, etype in enumerate(self.e2w):
            ys.append(self.proj[i](y))           # (batch_size, seq_len, dict_size)
        return ys
    

class MLM_remi(nn.Module):
    def __init__(self, e2w, emb_size, hidden_size):
        super().__init__()
        
        # proj: project embeddings to logits for prediction
        self.proj = nn.Linear(hidden_size, len(e2w))

        self.e2w = e2w
    
    def forward(self, y):
        # feed to bert 
        #y = y.last_hidden_state         # (batch_size, seq_len, 768)
        y = y.hidden_states[-1]

        # convert embeddings back to logits for prediction
        y = self.proj(y)           
        return y
