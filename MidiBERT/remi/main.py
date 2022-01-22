import argparse
import numpy as np
import random
import pickle
import os
import copy
import shutil
import json

from torch.utils.data import DataLoader
import torch
import torch.nn as nn
from transformers import BertConfig

from model import MidiBert
from trainer import BERTTrainer
from midi_dataset import MidiDataset

def get_args():
    parser = argparse.ArgumentParser(description='')

    ### path setup ###
    parser.add_argument('--dict_file', type=str, default='../../dict/remi.pkl')
    parser.add_argument('--name', type=str, default='')

    ### pre-train dataset ###
    parser.add_argument("--dataset", type=str, nargs='+', default=['pop909','composer', 'pop1k7', 'ASAP', 'emopia'])
    
    ### parameter setting ###
    parser.add_argument('--num_workers', type=int, default=5)
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--mask_percent', type=float, default=0.15, help="Up to `valid_seq_len * target_max_percent` tokens will be masked out for prediction")
    parser.add_argument('--max_seq_len', type=int, default=512, help='all sequences are padded to `max_seq_len`')
    parser.add_argument('--hs', type=int, default=768)
    parser.add_argument('--epochs', type=int, default=500, help='number of training epochs')
    parser.add_argument('--lr', type=float, default=2e-5, help='initial learning rate')
    
    ### cuda ###
    parser.add_argument("--cpu", action="store_true")   # default: true
    parser.add_argument("--cuda_devices", type=int, nargs='+', default=[0,1,2,3], help="CUDA device ids")

    args = parser.parse_args()

    return args


def load_data(datasets):
    to_concat = []
    root = '../../data/remi'

    for dataset in datasets:
        if dataset in {'pop909', 'composer', 'emopia'}:
            X_train = np.load(os.path.join(root, f'{dataset}_train.npy'), allow_pickle=True)
            X_valid = np.load(os.path.join(root, f'{dataset}_valid.npy'), allow_pickle=True)
            X_test = np.load(os.path.join(root, f'{dataset}_test.npy'), allow_pickle=True)
            data = np.concatenate((X_train, X_valid, X_test), axis=0)
            
        elif dataset == 'pop1k7' or dataset == 'ASAP':
            data = np.load(os.path.join(root, f'{dataset}.npy'), allow_pickle=True)

        print(f'   {dataset}: {data.shape}')
        to_concat.append(data)


    training_data = np.vstack(to_concat)
    print('   > all training data:', training_data.shape)
    
    # shuffle during training phase
    index = np.arange(len(training_data))
    np.random.shuffle(index)
    training_data = training_data[index]
    split = int(len(training_data)*0.85)
    X_train, X_val = training_data[:split], training_data[split:]
    
    return X_train, X_val


def main():
    args = get_args()

    print("Loading Dictionary")
    with open(args.dict_file, 'rb') as f:
        e2w, w2e = pickle.load(f)

    print("\nLoading Dataset") 
    X_train, X_val = load_data(set(args.dataset))
    
    trainset = MidiDataset(X=X_train)
    validset = MidiDataset(X=X_val) 

    train_loader = DataLoader(trainset, batch_size=args.batch_size, num_workers=args.num_workers, shuffle=True)
    print("   len of train_loader",len(train_loader))
    valid_loader = DataLoader(validset, batch_size=args.batch_size, num_workers=args.num_workers)
    print("   len of valid_loader",len(valid_loader))


    print("\nBuilding BERT model")
    configuration = BertConfig(max_position_embeddings=args.max_seq_len,
                                position_embedding_type='relative_key_query',
                                hidden_size=args.hs)

    midibert = MidiBert(bertConfig=configuration, e2w=e2w, w2e=w2e)

    print("\nCreating BERT Trainer")
    trainer = BERTTrainer(midibert, train_loader, valid_loader, args.lr, args.batch_size, args.max_seq_len, args.mask_percent, args.cpu, args.cuda_devices)
    
    
    print("\nTraining Start")
    save_dir = 'result/pretrain/' + args.name
    os.makedirs(save_dir, exist_ok=True)
    filename = os.path.join(save_dir, 'model.ckpt')
    print("   save model at {}".format(filename))

    best_acc, best_epoch = 0, 0
    bad_cnt = 0

    for epoch in range(args.epochs):
        if bad_cnt >= 30:
            print('valid acc not improving for 30 epochs')
            break
        train_loss, train_acc = trainer.train()
        print(train_loss, train_acc)
        valid_loss, valid_acc = trainer.valid()

        is_best = valid_acc > best_acc
        best_acc = max(valid_acc, best_acc)
        
        if is_best:
            bad_cnt, best_epoch = 0, epoch
        else:
            bad_cnt += 1
        
        print('epoch: {}/{} | Train Loss: {} | Train acc: {} | Valid Loss: {} | Valid acc: {}'.format(
            epoch, args.epochs, train_loss, train_acc, valid_loss, valid_acc))

        trainer.save_checkpoint(epoch, train_acc, valid_acc, 
                                valid_loss, train_loss, is_best, filename)


        with open(os.path.join(save_dir, 'log'), 'a') as outfile:
            outfile.write('Epoch {}: train_loss={}, train_acc={}, valid_loss={}, valid_acc={}\n'.format(
                epoch+1, train_loss, train_acc, valid_loss, valid_acc))


if __name__ == '__main__':
    main()
