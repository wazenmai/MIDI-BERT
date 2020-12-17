# -*- coding: utf-8 -*-
"""
Created on Dec 14 2020

@author: Yi-Hui (Sophia) Chou 
"""

from torch.utils.data import DataLoader
import tqdm
import torch
import torch.nn as nn
import utils_bestloss as utils
#import torch.nn.functional as F

def training(model, device, train_loader, optimizer):
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print('\nstart training, parameter total:{}, trainable:{}\n'.format(total, trainable))
    model.train()
    train_loss = 0
    pbar = tqdm.tqdm(train_loader, disable = False)
    for x, y in pbar:
        pbar.set_description("Training batch")
        x, y = x.to(device, dtype=torch.long), y.to(device, dtype=torch.float)
        #print("[train.py]",x.shape, y.shape)
        optimizer.zero_grad()
        
        y_hat = model(x) # this line could work if using gpu
        y_hat = y_hat.reshape((-1, 4))
        y_hat = y_hat.squeeze()
        y = y.reshape((-1)).to(dtype = torch.long) # required type of loss function
        
        # {-1: 5168130, 0: 309423, 1:247430, 2:976585}
        weights = [1, 5168130/309423, 5168130/247430, 5128130/976585]
        class_weights = torch.FloatTensor(weights).to(device)
        CE = nn.CrossEntropyLoss(weight = class_weights)
        loss = CE(y_hat, y)

        loss.backward()

        train_loss += loss
        optimizer.step()
    return train_loss/len(train_loader.dataset)

def valid(model, device, valid_loader):
    model.eval()
    valid_loss = 0
    with torch.no_grad():
        for x, y in valid_loader:
            x, y = x.to(device, dtype=torch.long), y.to(device, dtype=torch.float)
            
            y_hat = model(x) # beat activations (batch, timestep, 3) ==> nonbeat(0), donwbeat(1), beat(2)
            y_hat = y_hat.reshape((-1, 4))
            y = y.reshape((-1)).to(dtype = torch.long) # required type of loss function
            
            #weights = [1, 200, 67] # nonbeat, beat, downbeat
            weights = [1, 5168130/309423, 5168130/247430, 5128130/976585]
            class_weights = torch.FloatTensor(weights).to(device)
            CE = nn.CrossEntropyLoss(weight = class_weights)
            loss = CE(y_hat, y)
            valid_loss += loss
    return valid_loss/len(valid_loader.dataset)

