from torch.nn import LSTM, Linear  
import torch
import torch.nn as nn
import numpy as np
import math

class Embeddings(nn.Module):
    def __init__(self, n_token, d_model):
        super(Embeddings, self).__init__()
        self.lut = nn.Embedding(n_token, d_model)
        self.d_model = d_model

    def forward(self, x):
        return self.lut(x) * math.sqrt(self.d_model)


class LSTM_Net(nn.Module):
    # n_vocab: total number in dictionary.pkl
    def __init__(self, e2w, class_num, input_size=768, hidden_size=256, num_layers=3, dropout=0.5):
        super(LSTM_Net, self).__init__()
        self.e2w = e2w
        self.n_tokens = []      
        for key in e2w:
            self.n_tokens.append(len(e2w[key]))
        self.emb_sizes = [256, 256, 256, 256]

        # word_emb: embeddings to change token ids into embeddings
        self.word_emb = []
        for i, key in enumerate(e2w):
            self.word_emb.append(Embeddings(self.n_tokens[i], self.emb_sizes[i]))
        self.word_emb = nn.ModuleList(self.word_emb)

        #self.embeddings = nn.Embedding(vocab_size, input_size) 

        self.in_linear = nn.Linear(np.sum(self.emb_sizes), input_size)

        # proj: project embeddings to logits for prediction
        self.proj = []
        for i, etype in enumerate(e2w):
            self.proj.append(nn.Linear(hidden_size, self.n_tokens[i]))
        self.proj = nn.ModuleList(self.proj)

        self.lstm = LSTM(
            input_size = input_size, 
            hidden_size = hidden_size, 
            num_layers = num_layers, 
            bidirectional = True, 
            batch_first = True)

        self.classifier = nn.Sequential( nn.Dropout(dropout),
                                         nn.Linear(hidden_size*2, class_num+1).cuda() # class_num + <pad>
                                       ) 

    def forward(self, input_ids):
        # convert input_ids into embeddings and merge them through linear layer
        # input: (batch, seq, cp_token_num)
        embs = []
        for i, key in enumerate(self.e2w):
            embs.append(self.word_emb[i](input_ids[..., i]))
        embs = torch.cat([*embs], dim=-1)       # (batch, seq, 1024=256*4), where 256=emb_size
        emb_linear = self.in_linear(embs)   # 1024 to 128=hidden_size

        # feed to lstm
        y = self.lstm(emb_linear)   # (output, (h_n, c_n))'''
        y = self.classifier(y[0])   # y[0] = output
        return y
    
