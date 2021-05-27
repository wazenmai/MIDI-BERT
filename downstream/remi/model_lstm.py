# -*- coding: utf-8 -*-
"""
Created on Dec 14 2020 

@author: Yi-Hui (Sophia) Chou
"""

from torch.nn import LSTM, Linear, BatchNorm1d, Parameter
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import init

class LSTM_Net(nn.Module):
    # n_vocab: total number in dictionary.pkl
    def __init__(self, class_num, vocab_size=169, input_size=768, hidden_size=256, num_layers=3, dropout=0.4):
        super(LSTM_Net, self).__init__()

        self.embeddings = nn.Embedding(vocab_size, input_size) 
        self.lstm = LSTM(
            input_size = input_size, 
            hidden_size = hidden_size, 
            num_layers = num_layers, 
            bidirectional = True, 
            batch_first = True)

        self.classifier = nn.Sequential( nn.Dropout(dropout),
                                         nn.Linear(hidden_size*2, class_num+1).cuda()
                                        ) 
    def forward(self, x):
        x = self.embeddings(x)
        x = self.lstm(x)
        x = self.classifier(x[0])
        return x
