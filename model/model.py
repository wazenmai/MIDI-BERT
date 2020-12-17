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
    def __init__(self, vocab_size=271, input_size=271, hidden_size=128, num_layers=3, dropout=0.3):
        """
        input: (batch_size, seq_len, input_size)
        output: (batch_size, seq_len, 3)
        3: nonbeat, downbeat, beat activations """
        super(LSTM_Net, self).__init__()

        self.embeddings = nn.Embedding(vocab_size, input_size) 
        self.lstm = LSTM(
            input_size = input_size, 
            hidden_size = hidden_size, 
            num_layers = num_layers, 
            bidirectional = True, 
            batch_first = True)

        self.classifier = nn.Sequential( nn.Dropout(dropout),
                                         nn.Linear(hidden_size*2, 4).cuda(), # melody, bridge, piano
                                         nn.Softmax() )
    def forward(self, x):
        x = self.embeddings(x)
        #x, _ = self.lstm(inputs)
        # x's dimension: (batch, seq_len, hidden_size)
        # get the last hidden state of LSTM
        #x = x[:, -1, :] 
        #x = self.classifier(x)
        #print('[model.py] forward(): x.shape', x.shape)  #(16,512,271)
        x = self.lstm(x)
        x = self.classifier(x[0])
        return x
