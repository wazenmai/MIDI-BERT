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
from finetune_trainer import FinetuneTrainer
from finetune_dataset import FinetuneDataset

def get_args():
    parser = argparse.ArgumentParser(description='')

    ### mode ###
    parser.add_argument('--task', choices=['melody', 'velocity', 'composer', 'emotion'], required=True)
    ### path setup ###
    parser.add_argument('--dict_file', type=str, default='../dict/compact4/remi.pkl')
    parser.add_argument('--name', type=str, default='')
    parser.add_argument('--ans_path', type=str)

    ### pre-train dataset ###
    parser.add_argument("--dataset", type=str, choices=['pop909','composer', 'ailabs17k', 'ASAP', 'emopia'])
    
    ### parameter setting ###
    parser.add_argument('--num_workers', type=int, default=5)
    parser.add_argument('--class_num', type=int)
    parser.add_argument('--batch_size', type=int, default=12)
    parser.add_argument('--mask_percent', type=float, default=0.15, help="Up to `valid_seq_len * target_max_percent` tokens will be masked out for prediction")
    parser.add_argument('--max_seq_len', type=int, default=512, help='all sequences are padded to `max_seq_len`')
    parser.add_argument('--hs', type=int, default=768)
    parser.add_argument("--index_layer", type=int, default=12, help="number of layers")
    parser.add_argument("-l", "--layers", type=int, default=8, help="number of layers")
    parser.add_argument("-a", "--attn_heads", type=int, default=8, help="number of attention heads")
    parser.add_argument('--epochs', type=int, default=10, help='number of training epochs')
    parser.add_argument('--lr', type=float, default=2e-5, help='initial learning rate')
    parser.add_argument("--adam_weight_decay", type=float, default=0.01, help="weight_decay of adam")
    
    ### cuda ###
    parser.add_argument("--with_cuda", type=bool, default=True, help="training with CUDA: true, or false")
    parser.add_argument("--cuda_devices", type=int, nargs='+', default=[0,1,2,3], help="CUDA device ids")

    args = parser.parse_args()

    if args.task == 'melody':
        args.ans_path = '/home/yh1488/NAS-189/home/remi_data/POP909remi_melans.npy'
        args.class_num = 4
    elif args.task == 'velocity':
        args.ans_path = '/home/yh1488/NAS-189/home/remi_data/POP909remi_velans.npy'
        args.class_num = 7
    elif args.task == 'composer':
        args.class_num = 8
    elif args.task == 'emotion':
        args.class_num = 4

    return args


def load_data(dataset, answer):
    if dataset == 'pop909':
        POP909_path = '/home/yh1488/NAS-189/home/remi_data/POP909remi.npy'
        POP_data = np.load(POP909_path, allow_pickle=True)
        X_train, X_val, X_test = np.split(POP_data, [int(.8 * len(POP_data)), int(.9 * len(POP_data))])
        ans_data = np.load(answer, allow_pickle=True)
        y_train, y_val, y_test = np.split(ans_data, [int(.8 * len(ans_data)), int(.9 * len(ans_data))])

    elif dataset == 'composer':
        composer_root = '/home/yh1488/NAS-189/homes/wazenmai/datasets/MIDI-BERT/composer_dataset/new_remi/'
        X_train = np.load(composer_root+'CC_train_remi.npy', allow_pickle=True)
        X_val = np.load(composer_root+'CC_valid_remi.npy', allow_pickle=True)
        X_test = np.load(composer_root+'CC_test_remi.npy', allow_pickle=True)
        y_train = np.load(composer_root+'CC_train_ans_remi.npy', allow_pickle=True)
        y_val = np.load(composer_root+'CC_valid_ans_remi.npy', allow_pickle=True)
        y_test = np.load(composer_root+'CC_test_ans_remi.npy', allow_pickle=True)

    elif dataset == 'ailabs17k':
        remi1700_path = '/home/yh1488/NAS-189/home/remi_data/ai17k.npy'
        remi1700 = np.load(remi1700_path, allow_pickle=True)
        print('   ailabs17k:', remi1700.shape)

    elif dataset == 'emopia':
        emopia_root = '/home/yh1488/NAS-189/homes/wazenmai/datasets/MIDI-BERT/emopia_dataset/remi/emopia_'
        X_train = np.load(emopia_root+'train.npy', allow_pickle=True)
        X_val = np.load(emopia_root+'valid.npy', allow_pickle=True)
        X_test = np.load(emopia_root+'test.npy', allow_pickle=True)
        y_train = np.load(emopia_root+'train_ans.npy', allow_pickle=True)
        y_val = np.load(emopia_root+'valid_ans.npy', allow_pickle=True)
        y_test = np.load(emopia_root+'test_ans.npy', allow_pickle=True)
    else:
        print('dataset {} not supported'.format(dataset))
        exit(1)
    
    print('X_train: {}, X_valid: {}, X_test: {}'.format(
        X_train.shape, X_val.shape, X_test.shape))
    print('y_train: {}, y_valid: {}, y_test: {}'.format(
        y_train.shape, y_val.shape, y_test.shape))
    return X_train, X_val, X_test, y_train, y_val, y_test


def main():
    args = get_args()

    print("Loading Dictionary")
    with open(args.dict_file, 'rb') as f:
        e2w, w2e = pickle.load(f)

    print("\nLoading Dataset") 
    if args.task == 'melody' or args.task == 'velocity':
        dataset = 'pop909' 
        seq_class = False
    elif args.task == 'composer':
        dataset = 'composer'
        seq_class = True
    elif args.task == 'emotion':
        dataset = 'emopia'
        seq_class = True
    X_train, X_val, X_test, y_train, y_val, y_test = load_data(dataset, args.ans_path)
    
    trainset = FinetuneDataset(X=X_train, y=y_train)
    validset = FinetuneDataset(X=X_val, y=y_val) 
    testset = FinetuneDataset(X=X_test, y=y_test) 

    train_loader = DataLoader(trainset, batch_size=args.batch_size, num_workers=args.num_workers, shuffle=True)
    print("   len of train_loader",len(train_loader))
    valid_loader = DataLoader(validset, batch_size=args.batch_size, num_workers=args.num_workers)
    print("   len of valid_loader",len(valid_loader))
    test_loader = DataLoader(testset, batch_size=args.batch_size, num_workers=args.num_workers)
    print("   len of valid_loader",len(test_loader))


    print("\nBuilding BERT model")
    configuration = BertConfig(max_position_embeddings=args.max_seq_len,
                                position_embedding_type='relative_key_query',
                                hidden_size=args.hs)

    best_mdl = '/home/yh1488/NAS-189/homes/yh1488/BERT/remi_result/pretrain/model-bs12_best.ckpt'
    checkpoint = torch.load(best_mdl, map_location='cpu')
    midibert = MidiBert(bertConfig=configuration, e2w=e2w, w2e=w2e)
    midibert.load_state_dict(checkpoint['state_dict'])

    
    index_layer = int(args.index_layer)-13
    print("\nCreating Finetune Trainer using index layer", index_layer)
    trainer = FinetuneTrainer(midibert, train_loader, valid_loader, test_loader, index_layer, args.lr, args.class_num,
                                args.hs, y_test.shape, args.with_cuda, args.cuda_devices, None, seq_class)
    
    
    print("\nTraining Start")
    save_dir = '/home/yh1488/NAS-189/homes/yh1488/BERT/remi_result/finetune/'
    os.makedirs(save_dir, exist_ok=True)
    filename = save_dir + 'finetune-' + args.task + '-' + args.name + '.ckpt'
    print("   save model at {}".format(filename))

    best_acc, best_epoch = 0, 0
    bad_cnt = 0

    for epoch in range(args.epochs):
        if bad_cnt >= 3:
            print('valid acc not improving for 3 epochs')
            break
        train_loss, train_acc = trainer.train()
        valid_loss, valid_acc = trainer.valid()
        test_loss, test_acc, _ = trainer.test()

        is_best = valid_acc > best_acc
        best_acc = max(valid_acc, best_acc)
        
        if is_best:
            bad_cnt, best_epoch = 0, epoch
        else:
            bad_cnt += 1
        
        print('epoch: {}/{} | Train Loss: {} | Train acc: {} | Valid Loss: {} | Valid acc: {} | Test loss: {} | Test acc: {}'.format(
            epoch+1, args.epochs, train_loss, train_acc, valid_loss, valid_acc, test_loss, test_acc))

        trainer.save_checkpoint(epoch, train_acc, valid_acc, 
                                valid_loss, train_loss, is_best, filename)


        with open(os.path.join('log', args.task + '-' + args.name + '-finetune.log'), 'a') as outfile:
            outfile.write('Epoch {}: train_loss={}, valid_loss={}, test_loss={}, train_acc={}, valid_acc={}, test_acc={}\n'.format(
                epoch+1, train_loss, valid_loss, test_loss, train_acc, valid_acc, test_acc))

if __name__ == '__main__':
    main()
