import math
import numpy as np
import random

import torch
import torch.nn as nn

from model import MidiBert


class FinetuneModel(nn.Module):
    def __init__(self, midibert: MidiBert, class_num, hs):
        super().__init__()
        
        self.midibert = midibert
        self.finetune = Finetune(class_num, hs)

    def forward(self, x, attn):
        x = self.midibert(x, attn)
        return self.finetune(x)
    

class Finetune(nn.Module):
    def __init__(self, class_num, hs):
        super().__init__()
        
        self.dropout = nn.Dropout(0.1)
        self.proj = nn.Linear(hs, 256)
        self.relu =  nn.ReLU()
        self.output = nn.Linear(256, class_num)
    
    def forward(self, y):
        # feed to bert 
        #y = y.last_hidden_state         # (batch_size, seq_len, 768)

        y = self.dropout(y)
        y = self.proj(y)
        y = self.relu(y)
        return self.output(y)
