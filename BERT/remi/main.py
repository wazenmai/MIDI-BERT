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
    parser.add_argument('--dict_file', type=str, default='../dict/remi.pkl')
    parser.add_argument('--name', type=str, default='')

    ### pre-train dataset ###
    parser.add_argument("--dataset", type=str, nargs='+', default=['pop909','composer', 'ailabs17k', 'ASAP', 'emopia'])
    
    ### parameter setting ###
    parser.add_argument('--num_workers', type=int, default=5)
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--mask_percent', type=float, default=0.15, help="Up to `valid_seq_len * target_max_percent` tokens will be masked out for prediction")
    parser.add_argument('--max_seq_len', type=int, default=512, help='all sequences are padded to `max_seq_len`')
    parser.add_argument('--hs', type=int, default=768)
    parser.add_argument("-l", "--layers", type=int, default=12, help="number of layers")
    parser.add_argument("-a", "--attn_heads", type=int, default=12, help="number of attention heads")
    parser.add_argument('--epochs', type=int, default=500, help='number of training epochs')
    parser.add_argument('--lr', type=float, default=2e-5, help='initial learning rate')
    parser.add_argument("--adam_weight_decay", type=float, default=0.01, help="weight_decay of adam")
    
    ### cuda ###
    parser.add_argument("--with_cuda", type=bool, default=True, help="training with CUDA: true, or false")
    parser.add_argument("--cuda_devices", type=int, nargs='+', default=[0,2,3], help="CUDA device ids")
    #parser.add_argument("-w", "--num_workers", type=int, default=5, help="dataloader worker size")

    args = parser.parse_args()

    return args


def load_data(dataset):
    to_concat = []
    
    if 'pop909' in dataset:
        POP909_path = '/home/yh1488/NAS189/home/remi_data/POP909remi.npy'
        POP_data = np.load(POP909_path, allow_pickle=True)
        print('   POP909:', POP_data.shape)
        to_concat.append(POP_data)

    if 'composer' in dataset:
        composer_root = '/home/yh1488/NAS189/homes/wazenmai/datasets/MIDI-BERT/composer_dataset/new_remi/'
        path = composer_root + 'CC_train_remi.npy'
        composer_train = np.load(path, allow_pickle=True)
        path = composer_root + 'CC_valid_remi.npy'
        composer_valid = np.load(path, allow_pickle=True)
        path = composer_root + 'CC_test_remi.npy'
        composer_test = np.load(path, allow_pickle=True)
        composer = np.concatenate((composer_train, composer_valid, composer_test), axis=0)
        print('   composer:', composer.shape)
        to_concat.append(composer)

    if 'ailabs17k' in dataset:
        remi1700_path = '/home/yh1488/NAS189/home/remi_data/ai17k.npy'
        remi1700 = np.load(remi1700_path, allow_pickle=True)
        print('   ailabs17k:', remi1700.shape)
        to_concat.append(remi1700)

    if 'ASAP' in dataset:
        ASAP_path = '/home/yh1488/NAS189/homes/wazenmai/MIDI-BERT/for_wazenmai/prepare_remi/ASAP_remi.npy'
        ASAP = np.load(ASAP_path, allow_pickle=True)
        print('   ASAP:', ASAP.shape)
        to_concat.append(ASAP)

    if 'emopia' in dataset:
        emopia_root = '/home/yh1488/NAS189/homes/wazenmai/datasets/MIDI-BERT/emopia_dataset/remi/emopia_'
        path = emopia_root + 'train.npy'
        emopia_train = np.load(path, allow_pickle=True)
        path = emopia_root + 'valid.npy'
        emopia_valid = np.load(path, allow_pickle=True)
        path = emopia_root + 'test.npy'
        emopia_test = np.load(path, allow_pickle=True)
        emopia = np.concatenate((emopia_train, emopia_valid, emopia_test), axis=0)
        print('   emopia:', emopia.shape)
        to_concat.append(emopia)
    
    #training_data = np.concatenate((composer_train, remi1700, composer_valid, POP_data, composer_test, ASAP, emopia), axis=0) 
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
    trainer = BERTTrainer(midibert, train_loader, valid_loader, args.lr, args.batch_size, args.max_seq_len, args.mask_percent, args.with_cuda, args.cuda_devices)
    
    
    print("\nTraining Start")
    save_dir = '/home/yh1488/NAS189/homes/yh1488/BERT/remi_result/pretrain/'
    os.makedirs(save_dir, exist_ok=True)
    filename = save_dir + 'model-' + args.name + '.ckpt'
    print("   save model at {}".format(filename))

    best_acc, best_epoch = 0, 0
    bad_cnt = 0

    for epoch in range(args.epochs):
        if bad_cnt >= 20:
            print('valid acc not improving for 20 epochs')
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


        with open(os.path.join('log', args.name + '.log'), 'a') as outfile:
            outfile.write('Epoch {}: train_loss={}, valid_loss={}, train_acc={}, valid_acc={}\n'.format(
                epoch, train_loss, valid_loss, train_acc, valid_acc))
            #outfile.write(json.dumps(log, indent=4, sort_keys=True))

if __name__ == '__main__':
    main()
