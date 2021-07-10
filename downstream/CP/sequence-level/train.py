# -*- coding: utf-8 -*-
"""
Created on Dec 14 2020
@author: Yi-Hui (Sophia) Chou 

Updated on May 10 2021
@author: I-Chun (Bronwin) Chen
"""
import sys
sys.path.append('../../CP')

import tqdm
import torch
import numpy as np
import torch.nn as nn
import utils_bestloss as utils
from torch.utils.data import DataLoader
from sklearn.metrics import confusion_matrix

def training(model, device, train_loader, optimizer, batch, class_num):
    model.train()
    
    train_loss = 0 
    pbar = tqdm.tqdm(train_loader, disable = False)

    for x, y in pbar:
        pbar.set_description("Training batch")
        x, y = x.to(device).long(), y.to(device).long()
        optimizer.zero_grad()
        
        y_hat = model(x)
        y_hat = y_hat.reshape((-1, class_num))
        y_hat = y_hat.squeeze()
        y = y.reshape((-1)).to(dtype = torch.long)
        
        _, pred_label = torch.max(y_hat, dim=1)
        y_np = y.detach().cpu().numpy()
        pred_np = pred_label.detach().cpu().numpy()

        cm = confusion_matrix(y_np, pred_np, labels=[x for x in range(class_num)])
        acc, sum = 0, 0
        for i in range(0, class_num):
            acc += cm[i][i]
            for j in range(0, class_num):
                sum += cm[i][j]        
        accuracy = acc/sum

        loss_func = nn.CrossEntropyLoss(reduction='mean')
        loss = loss_func(y_hat, y)
        loss.backward()

        train_loss += loss
        optimizer.step()

    return train_loss/len(train_loader), accuracy


def valid(model, device, valid_loader, batch, class_num):
    model.eval()
    valid_loss, valid_acc = 0, 0
    with torch.no_grad():
        for x, y in valid_loader:
            x, y = x.to(device).long(), y.to(device).long()
            y_hat = model(x) 
            y_hat = y_hat.reshape((-1, class_num))
            y = y.reshape((-1)).to(dtype = torch.long)

            _, pred_label = torch.max(y_hat, dim=1)
            y_np = y.detach().cpu().numpy()
            pred_np = pred_label.detach().cpu().numpy()

            cm = confusion_matrix(y_np, pred_np, labels=[x for x in range(class_num)])
            acc, sum = 0, 0
            for i in range(0,class_num):
                acc += cm[i][i]
                for j in range(0,class_num):
                    sum += cm[i][j]        
            accuracy = acc/sum

            loss_func = nn.CrossEntropyLoss(reduction='mean')
            loss = loss_func(y_hat, y)

            valid_loss += loss

    return valid_loss/len(valid_loader), accuracy
