# -*- coding: utf-8 -*-
"""
Created on Dec 14 2020
@author: Yi-Hui (Sophia) Chou 

Updated on May 10 2021
@author: I-Chun (Bronwin) Chen
"""
import sys
sys.path.append('../../CP')

import os
import json
import time
import tqdm
import torch
import pickle
import argparse
import numpy as np
import torch.nn as nn
import utils_bestloss as utils

from pathlib import Path
from model import SAN
from train import training, valid
from pop_dataset import PopDataset
from torch.utils.data import DataLoader

def get_args():
    parser = argparse.ArgumentParser(description='Argument Parser for sequence-level tasks')

    ### mode ###
    parser.add_argument('--task', choices=['composer', 'emotion'], required=True)

    ### path setup ### 
    parser.add_argument('--input', type=str, default='../../../data/CP',help='Path to input numpy folder for composer dataset')
    parser.add_argument('--dict', type=str, default='../../../BERT/dict/CP.pkl')
    parser.add_argument('--output', type=str, help='Used for output directory name', required=True)
    
    ### parameter setting ###
    parser.add_argument('--train-batch', default=16, type=int)
    parser.add_argument('--dev-batch', default=8, type=int)
    parser.add_argument('--cuda', default=0, type=int, help='Specify cuda number')
    parser.add_argument('--epoch', default=1000, type=int, help='number of training epochs')
    parser.add_argument('--lr', default=1e-2, type=float, help="learning rate")
    
    args = parser.parse_args()

    # learning rate used in paper
    # if args.lr == 0:
    #     if args.task == "composer":
    #         args.lr = 1e-2
    #     elif args.task == "emotion":
    #         args.lr = 5e-2
    
    if args.task == "composer":
        args.num_of_class = 8
    elif args.task == "emotion":
        args.num_of_class = 4

    return args

def main():
    args = get_args()
    cuda_num = args.cuda 
    cuda_str = 'cuda:'+str(cuda_num)
    device = torch.device(cuda_str if torch.cuda.is_available() else 'cpu')

    inputs = args.input
    exp_name = args.output
    exp_dir = os.path.join('./experiments', exp_name)
    target_jsonpath = exp_dir
    num_of_class = args.num_of_class

    train_epochs = args.epoch
    lr = args.lr
    patience = 20
    
    if not os.path.exists(exp_dir):
        Path(exp_dir).mkdir(parents = True, exist_ok = True)

    print("loading dictionary...")
    with open(args.dict, 'rb') as f:
        e2w, w2e = pickle.load(f)

    print("\nloading data...")
    if args.task == "composer":
        X_train = torch.tensor(np.load(inputs + "/composer_train.npy", allow_pickle=True), dtype=torch.long)
        X_val = torch.tensor(np.load(inputs + "/composer_valid.npy", allow_pickle=True), dtype=torch.long)
        y_train = torch.tensor(np.load(inputs+ "/composer_train_ans.npy", allow_pickle=True), dtype=torch.long)
        y_val = torch.tensor(np.load(inputs+ "/composer_valid_ans.npy", allow_pickle=True), dtype=torch.long)
    elif args.task == "emotion":
        X_train = torch.tensor(np.load(inputs + "/emopia_train.npy", allow_pickle=True), dtype=torch.long)
        X_val = torch.tensor(np.load(inputs + "/emopia_valid.npy", allow_pickle=True), dtype=torch.long)
        y_train = torch.tensor(np.load(inputs+ "/emopia_train_ans.npy", allow_pickle=True), dtype=torch.long)
        y_val = torch.tensor(np.load(inputs+ "/emopia_valid_ans.npy", allow_pickle=True), dtype=torch.long)
    trainset = PopDataset(X=X_train, y=y_train)
    validset = PopDataset(X=X_val, y=y_val)
    train_loader = DataLoader(trainset, batch_size = args.train_batch, shuffle = True)
    print("   len of train_loader", len(train_loader))
    valid_loader = DataLoader(validset, batch_size = args.dev_batch, shuffle = True)
    print("   len of valid_loader", len(valid_loader))
    print("\n initializing model...")    

    model = SAN(num_of_dim=num_of_class, e2w=e2w, vocab_size=len(e2w), embedding_size=768, r=4)
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
    train_losses, train_accs = [], []
    valid_losses, valid_accs = [], []
    train_times = []
    best_epoch = 0
    stop_t = 0
    print("   start training...")

    for epoch in t:
        # break
        t.set_description("Training Epoch")
        end = time.time()
        train_loss, train_acc = training(model, device, train_loader, optimizer, args.train_batch, num_of_class)
        valid_loss, valid_acc = valid(model, device, valid_loader, args.dev_batch, num_of_class)
        
        scheduler.step(valid_loss)
        train_losses.append(train_loss.item())
        valid_losses.append(valid_loss.item())
        train_accs.append(train_acc.item())
        valid_accs.append(valid_acc.item())

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
            target='SAN_' + args.task
        )

        # save params
        params = {
            'epochs_trained': epoch,
            'best_loss': es.best,
            'best_epoch': best_epoch,
            'train_loss_history': train_losses,
            'valid_loss_history': valid_losses,
            'train_acc_history': train_accs,
            'valid_acc_history': valid_accs,
            'train_time_history': train_times,
            'num_bad_epochs': es.num_bad_epochs,
        }

        with open(os.path.join(target_jsonpath, 'SAN_' + args.task + '.json'), 'w') as outfile:
            outfile.write(json.dumps(params, indent=4, sort_keys=True))

        train_times.append(time.time() - end)
        
        if stop:
            print("Apply Early Stopping and retrain")
            # break
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
