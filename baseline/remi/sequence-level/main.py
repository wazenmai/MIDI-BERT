# -*- coding: utf-8 -*-
"""
Created on Dec 14 2020

@author: Yi-Hui (Sophia) Chou 
"""
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

from model import SAN
from pathlib import Path
from train import training, valid
from pop_dataset import PopDataset
from torch.utils.data import DataLoader

def get_args():
    parser = argparse.ArgumentParser(description='Argument Parser for sequence-level baseline')

    ### mode ###
    parser.add_argument('--task', choices=['composer', 'emotion'], required=True)

    ### path setup ###
    parser.add_argument('--input', type=str, default='../../../data/remi', help='Path to input numpy file for pop909 dataset')
    parser.add_argument('--dict', type=str, default="../../../BERT/dict/remi.pkl", help='Path to dictionary of event')
    parser.add_argument('--output', type=str, help='Used for output directory name', required=True)
    
    ### parameter setting ###
    parser.add_argument('--train-batch', default=16, type=int)
    parser.add_argument('--dev-batch', default=8, type=int)
    parser.add_argument('--cuda', default=1, type=int, help='Specify cuda number')
    parser.add_argument('--epoch', default=1000, type=int, help='number of training epochs')
    parser.add_argument('--patience', default=20, type=int, help='patience number')
    parser.add_argument('--lr', default=5e-2, type=float, help='learning rate')

    args = parser.parse_args()

    if args.task == 'composer':
        args.class_num = 8
    elif args.task == 'emotion':
        args.class_num = 4

    return args

def load_data(task):
    if task == "composer":
        X_train = torch.tensor(np.load(inputs + "/composer_remi_train.npy", allow_pickle=True), dtype=torch.long)
        X_val = torch.tensor(np.load(inputs + "/composer_remi_valid.npy", allow_pickle=True), dtype=torch.long)
        y_train = torch.tensor(np.load(inputs+ "/composer_remi_train_ans.npy", allow_pickle=True), dtype=torch.long)
        y_val = torch.tensor(np.load(inputs+ "/composer_remi_valid_ans.npy", allow_pickle=True), dtype=torch.long)
    elif task == "emotion":
        X_train = torch.tensor(np.load(inputs + "/emopia_remi_train.npy", allow_pickle=True), dtype=torch.long)
        X_val = torch.tensor(np.load(inputs + "/emopia_remi_valid.npy", allow_pickle=True), dtype=torch.long)
        y_train = torch.tensor(np.load(inputs+ "/emopia_remi_train_ans.npy", allow_pickle=True), dtype=torch.long)
        y_val = torch.tensor(np.load(inputs+ "/emopia_remi_valid_ans.npy", allow_pickle=True), dtype=torch.long)

def main():
    args = get_args()
    cuda_num = args.cuda 
    cuda_str = 'cuda:'+str(cuda_num)
    device = torch.device(cuda_str if torch.cuda.is_available() else 'cpu')
    # device = 'cpu'

    inputs = args.input
    dict_name = args.dict
    exp_name = args.output
    exp_dir = os.path.join('./experiments', exp_name)
    target_jsonpath = exp_dir

    train_epochs = args.epoch
    lr = args.lr
    patience = args.patience
    
    if not os.path.exists(exp_dir):
        Path(exp_dir).mkdir(parents = True, exist_ok = True)

    print("loading data...")
    X_train, X_val, y_train, y_val = load_data(args.task)

    event2word, word2event = pickle.load(open(dict_name, 'rb'))
    
    trainset = PopDataset(X=X_train, y=y_train)
    validset = PopDataset(X=X_val, y=y_val) # can use different txt to initialize
    train_loader = DataLoader(trainset, batch_size = args.train_batch, shuffle = True, drop_last=True)
    print("len of train_loader",len(train_loader))
    valid_loader = DataLoader(validset, batch_size = args.dev_batch, shuffle = True, drop_last=True)
    print("len of valid_loader",len(valid_loader))

    print("initializing model...")  
    model = SAN(num_of_dim=args.class_num, vocab_size=len(event2word), embedding_size=768, r=4) # LSTM_Net(e2w=event2word)
    model.cuda(cuda_num)

    optimizer = torch.optim.SGD(
        model.parameters(),
        lr=lr,
        momentum=0.9
    )

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        factor=0.3,
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
    print("start training...")

    for epoch in t:
        t.set_description("Training Epoch")
        end = time.time()
        train_loss, train_acc = training(model, device, train_loader, optimizer, args.class_num)
        valid_loss, valid_acc = valid(model, device, valid_loader, args.class_num)
        
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

        with open(os.path.join(target_jsonpath,  'SAN_' + args.task + '.json'), 'w') as outfile:
            outfile.write(json.dumps(params, indent=4, sort_keys=True))

        train_times.append(time.time() - end)
        
        if stop:
            print("Apply Early Stopping and retrain")
            # break
            stop_t += 1
            if stop_t >= 5:
                break
            lr = lr * 0.2
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

