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
    parser.add_argument('--dict_file', type=str, default='../dict/compact4/CP.pkl')
    parser.add_argument('--name', type=str, default='')
    parser.add_argument('--ans_path', type=str)

    ### pre-train dataset ###
    parser.add_argument("--dataset", type=str, choices=['pop909','composer', 'ailabs17k', 'ASAP', 'emopia'])
    
    ### parameter setting ###
    parser.add_argument('--num_workers', type=int, default=5)
    parser.add_argument('--class_num')
    parser.add_argument('--batch_size', type=int, default=12)
    parser.add_argument('--mask_percent', type=float, default=0.15, help="Up to `valid_seq_len * target_max_percent` tokens will be masked out for prediction")
    parser.add_argument('--max_seq_len', type=int, default=512, help='all sequences are padded to `max_seq_len`')
    parser.add_argument('--hs', type=int, default=768)
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
        args.ans_path = '/home/yh1488/NAS-189/home/CP_data/POP909cp_melans.npy'
        args.class_num = 4
    elif args.task == 'velocity':
        args.ans_path = '/home/yh1488/NAS-189/home/CP_data/POP909cp_velans.npy'
        args.class_num = 7
    elif args.task == 'composer':
        pass
    elif args.task == 'emotion':
        pass

    return args


def load_data(dataset):
    if dataset == 'pop909':
        POP909_path = '/home/yh1488/NAS-189/home/CP_data/POP909cp.npy'
        POP_data = np.load(POP909_path, allow_pickle=True)
        X_train, X_val, X_test = np.split(POP_data, [int(.8 * len(POP_data)), int(.9 * len(POP_data))])

    elif dataset == 'composer':
        composer_path = '/home/yh1488/NAS-189/homes/wazenmai/datasets/MIDI-BERT/composer_dataset/CP/composer_cp_train.npy'
        composer_train = np.load(composer_path, allow_pickle=True)
        composer_path = '/home/yh1488/NAS-189/homes/wazenmai/datasets/MIDI-BERT/composer_dataset/CP/composer_cp_valid.npy'
        composer_valid = np.load(composer_path, allow_pickle=True)
        composer_path = '/home/yh1488/NAS-189/homes/wazenmai/datasets/MIDI-BERT/composer_dataset/CP/composer_cp_test.npy'
        composer_test = np.load(composer_path, allow_pickle=True)
        print('   Composer: ({}, {}, {})'.format(composer_train.shape[0]+composer_valid.shape[0]+composer_test.shape[0], composer_train.shape[1], composer_train.shape[2]))

    elif dataset == 'ailabs17k':
        remi1700_path = '/home/yh1488/NAS-189/home/CP_data/ai17k.npy'
        remi1700 = np.load(remi1700_path, allow_pickle=True)
        print('   ailabs17k:', remi1700.shape)

    elif dataset == 'ASAP':
        ASAP_path = '/home/yh1488/NAS-189/homes/wazenmai/MIDI-BERT/for_wazenmai/prepare_CP/ASAP_CP.npy'
        ASAP = np.load(ASAP_path, allow_pickle=True)
        print('   ASAP:', ASAP.shape)

    elif dataset == 'emopia':
        emopia_path = '/home/yh1488/NAS-189/homes/wazenmai/MIDI-BERT/for_wazenmai/prepare_CP/Emopia_CP.npy'
        emopia = np.load(emopia_path, allow_pickle=True)
        print('   emopia:', emopia.shape)
    
    
    print('train', X_train.shape)
    print('valid', X_val.shape)
    print('test', X_test.shape)
    return X_train, X_val, X_test


def main():
    args = get_args()

    print("Loading Dictionary")
    with open(args.dict_file, 'rb') as f:
        e2w, w2e = pickle.load(f)

    print("\nLoading Dataset") 
    if args.task == 'melody' or args.task == 'velocity':
        dataset = 'pop909' 
    X_train, X_val, X_test = load_data(dataset)
    ans_data = np.load(args.ans_path, allow_pickle=True)
    y_train, y_val, y_test = np.split(ans_data, [int(.8 * len(ans_data)), int(.9 * len(ans_data))])
    
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

    best_mdl = '/home/yh1488/NAS-189/homes/yh1488/BERT/cp_result/pretrain/model-batch12_best.ckpt'
    checkpoint = torch.load(best_mdl, map_location='cpu')
    midibert = MidiBert(bertConfig=configuration, e2w=e2w, w2e=w2e)
    midibert.load_state_dict(checkpoint['state_dict'])

    
    print("\nCreating Finetune Trainer")
    trainer = FinetuneTrainer(midibert, train_loader, valid_loader, test_loader, args.lr, args.class_num,
                                args.hs, args.with_cuda, args.cuda_devices)
    
    
    print("\nTraining Start")
    save_dir = '/home/yh1488/NAS-189/homes/yh1488/BERT/cp_result/finetune/'
    os.makedirs(save_dir, exist_ok=True)
    filename = save_dir + 'finetune-' + args.name + '.ckpt'
    print("   save model at {}".format(filename))

    best_acc, best_epoch = 0, 0
    bad_cnt = 0

    for epoch in range(args.epochs):
        if bad_cnt >= 3:
            print('valid acc not improving for 3 epochs')
            break
        train_loss, train_acc = trainer.train()
        valid_loss, valid_acc = trainer.valid()
        test_loss, test_acc = trainer.test()

        is_best = valid_acc > best_acc
        best_acc = max(valid_acc, best_acc)
        
        if is_best:
            bad_cnt, best_epoch = 0, epoch
        else:
            bad_cnt += 1
        
        print('epoch: {}/{} | Train Loss: {} | Train acc: {} | Valid Loss: {} | Valid acc: {} | Test loss: {} | Test acc: {}'.format(
            epoch, args.epochs, train_loss, train_acc, valid_loss, valid_acc, test_loss, test_acc))

        trainer.save_checkpoint(epoch, train_acc, valid_acc, 
                                valid_loss, train_loss, is_best, filename)


        with open(os.path.join('log', args.task + '-' + args.name + '-finetune.log'), 'a') as outfile:
            outfile.write('Epoch {}: train_loss={}, valid_loss={}, test_loss={}, train_acc={}, valid_acc={}, test_acc={}\n'.format(
                epoch, train_loss, valid_loss, test_loss, train_acc, valid_acc, test_acc))
            #outfile.write(json.dumps(log, indent=4, sort_keys=True))

if __name__ == '__main__':
    main()
