import sys
sys.path.append('../../CP')

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
import argparse


def get_args():
    parser = argparse.ArgumentParser(description='Argument Parser for downstream classification')
    ### mode ###
    parser.add_argument('--task', choices=['melody', 'velocity'], required=True)
    
    ### path setup ### 
    parser.add_argument('--name', type=str, help='Used for output directory name', required=True)
    
    ### parameter setting ###
    parser.add_argument('--train-batch', default=16, type=int)
    parser.add_argument('--valid-batch', default=8, type=int)
    parser.add_argument('--cuda', default=0, type=int, help='Specify cuda number')
    parser.add_argument('--epoch', default=500, type=int)
    parser.add_argument('--patience', default=20, type=int)
    parser.add_argument('--lr', type=float, default=1e-3)

    args = parser.parse_args()
    
    if args.task == 'melody':
        args.class_num = 3
    elif args.task == 'velocity':
        args.class_num = 6 
    
    return args


def load_data(task):
    task_name = 'melody identification' if task == 'melody' else 'velocity classification'
    print("\nloading data for {} ...".format(task_name))
    root = '../../data/remi/pop909_'
    X_train = np.load(root+'train.npy', allow_pickle=True)
    X_val = np.load(root+'valid.npy', allow_pickle=True)
    y_train = np.load(root+'train_'+task[:3]+'ans.npy', allow_pickle=True)
    y_val = np.load(root+'valid_'+task[:3]+'ans.npy', allow_pickle=True)
    
    return X_train, X_val, y_train, y_val


def main():
    args = get_args()
    cuda_num = args.cuda 
    cuda_str = 'cuda:'+str(cuda_num)
    device = torch.device(cuda_str if torch.cuda.is_available() else 'cpu')
    print(device)

    exp_dir = os.path.join('result', args.task + '-LSTM', args.name)
    target_jsonpath = exp_dir
    if not os.path.exists(exp_dir):
        Path(exp_dir).mkdir(parents = True, exist_ok = True)

    train_epochs = args.epoch 
    lr = args.lr
    patience = args.patience
    
    X_train, X_val, y_train, y_val = load_data(args.task)
        
    print('train shape', X_train.shape,'; valid shape', X_val.shape)
    trainset = PopDataset(X=X_train, y=y_train)
    validset = PopDataset(X=X_val, y=y_val) # can use different txt to initialize
    train_loader = DataLoader(trainset, batch_size = args.train_batch, shuffle = True, drop_last = False)#, drop_last=True)
    print("len of train_loader",len(train_loader))
    valid_loader = DataLoader(validset, batch_size = args.valid_batch, shuffle = True, drop_last = False)#, drop_last=True)
    print("len of valid_loader",len(valid_loader))
    
    print("initializing model...")    
    model = LSTM_Net(class_num=args.class_num)
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
    print("start training...")

    for epoch in t:
#        break
        t.set_description("Training Epoch")
        end = time.time()
        train_loss, train_acc = training(model, device, train_loader, optimizer, args.train_batch, args.class_num+1)
        valid_loss, valid_acc = valid(model, device, valid_loader, args.valid_batch, args.class_num+1)
        
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
                target='LSTM-'+args.task+'-classification'
            )

            # save params
        params = {
                'epochs_trained': epoch,
#                'args': vars(args),
                'best_loss': es.best,
                'best_epoch': best_epoch,
                'train_loss_history': train_losses,
                'valid_loss_history': valid_losses,
                'train_acc_history': train_accs,
                'valid_acc_history': valid_accs,
                'train_time_history': train_times,
                'num_bad_epochs': es.num_bad_epochs,
    #            'commit': commit
            }

        with open(os.path.join(target_jsonpath,  'LSTM-' + args.task + '.json'), 'w') as outfile:
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
