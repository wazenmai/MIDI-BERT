# -*- coding: utf-8 -*-
"""
Created on Dec 14 2020

@author: Yi-Hui (Sophia) Chou 
"""
import os
from model import LSTM_Net 
from pop_dataset import PopDataset
from torch.utils.data import DataLoader
import tqdm
import torch
import torch.nn as nn
from pathlib import Path
import json
import utils_bestloss as utils
import time
from train import training, valid
import numpy as np
import sys
#import torch.nn.functional as F

def main():
    cuda_num = 1 
    cuda_str = 'cuda:'+str(cuda_num)
    device = torch.device(cuda_str if torch.cuda.is_available() else 'cpu')
    train_epochs = 500
    # must assign
    date = sys.argv[1]
    if len(sys.argv)!=2:
        print('Usage: python3 main.py [date]')
        exit(1)
    exp_name = 'LSTM_pop909_'+date
    exp_dir = os.path.join('./experiments', exp_name)
    target_jsonpath = exp_dir
    lr = 1e-3
    patience = 20
    
    if not os.path.exists(exp_dir):
        Path(exp_dir).mkdir(parents = True, exist_ok = True)

    print("loading data...")
    #all = np.load('/home/yh1488/remi/remi909/pop909all.npy')
    all = np.load('./pop909all.npy')
    all = torch.tensor(all, dtype=torch.long)
    #all_ans = np.load('/home/yh1488/remi/remi909/pop909all_ans.npy')
    all_ans = np.load('./pop909all_ans.npy')
    all_ans = torch.tensor(all_ans, dtype=torch.long)
    # prepare data to 8:2 (10470) 
    X_train, X_val, y_train, y_val = all[:10470], all[10470:], all_ans[:10470], all_ans[10470:]
    
    trainset = PopDataset(X=X_train, y=y_train)
    validset = PopDataset(X=X_val, y=y_val) # can use different txt to initialize
    train_loader = DataLoader(trainset, batch_size = 16, shuffle = True)
    valid_loader = DataLoader(validset, batch_size = 8, shuffle = True)
    print("initializing model...")    
    model = LSTM_Net()
    model.cuda(cuda_num)

    optimizer = torch.optim.SGD(
            model.parameters(),
            lr=lr,
            momentum=0.9
        )

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            factor=0.3, # follow openunmix
            patience=80,
            cooldown=10
        )

    es = utils.EarlyStopping(patience= patience)
    
    t = tqdm.trange(1, train_epochs +1, disable = False)
    train_losses = []
    valid_losses = []
    train_times = []
    best_epoch = 0
    stop_t = 0
    print("start training...")

    for epoch in t:
#        break
        t.set_description("Training Epoch")
        end = time.time()
        train_loss = training(model, device, train_loader, optimizer)
        valid_loss = valid(model, device, valid_loader)
        
        scheduler.step(valid_loss)
        train_losses.append(train_loss.item())
        valid_losses.append(valid_loss.item())

        t.set_postfix(
        train_loss=train_loss.item(), val_loss=valid_loss.item()
        )

        stop = es.step(valid_loss.item())

        if valid_loss.item() == es.best:
            best_epoch = epoch
            
        utils.save_checkpoint({
                    'epoch': epoch + 1,
                    'state_dict': model.state_dict(),
                    'best_loss': es.best,
                    'optimizer': optimizer.state_dict(),
                    'scheduler': scheduler.state_dict()
                },
                is_best=valid_loss == es.best,
                path=exp_dir,
                target='RNNBeatProc'
            )

            # save params
        params = {
                'epochs_trained': epoch,
#                'args': vars(args),
                'best_loss': es.best,
                'best_epoch': best_epoch,
                'train_loss_history': train_losses,
                'valid_loss_history': valid_losses,
                'train_time_history': train_times,
                'num_bad_epochs': es.num_bad_epochs,
    #            'commit': commit
            }

        with open(os.path.join(target_jsonpath,  'RNNbeat' + '.json'), 'w') as outfile:
            outfile.write(json.dumps(params, indent=4, sort_keys=True))

        train_times.append(time.time() - end)
        
        if stop:
                print("Apply Early Stopping and retrain")
#                break
                stop_t +=1
                if stop_t >=5:
                    break
                lr = lr*0.2
                optimizer = torch.optim.SGD(
                    model.parameters(),
                    lr=lr,
                    momentum=0.9
                )

                scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                        optimizer,
                        factor=0.3, # follow openunmix
                        patience=80,
                        cooldown=10
                    )

                es = utils.EarlyStopping(patience= patience, best_loss = es.best)
                
            
if __name__ == "__main__":
    main()
