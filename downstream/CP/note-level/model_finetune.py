# -*- coding: utf-8 -*-
"""
Created on Dec 14 2020 

@author: Yi-Hui (Sophia) Chou
"""

from torch.nn import LSTM, Linear  
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.nn import init
import math

class LSTM_Finetune(nn.Module):
    # n_vocab: total number in dictionary.pkl
    def __init__(self, class_num, hidden_size, input_size=768, num_layers=3, dropout=0.5):
        super(LSTM_Finetune, self).__init__()

        self.lstm = LSTM(
            input_size = input_size, 
            hidden_size = hidden_size, 
            num_layers = num_layers, 
            bidirectional = True, 
            batch_first = True)

        self.classifier = nn.Sequential( nn.Dropout(dropout),
                                         nn.Linear(hidden_size*2, class_num+1).cuda()
                                       ) 
    def forward(self, input_embed):
        # feed to lstm
        y = self.lstm(input_embed)
        # observed output: (batch, seq, hidden*2)
        y = self.classifier(y[0])   # y[0] = output
        return y   # (batch, seq_len, class_probability)

    
