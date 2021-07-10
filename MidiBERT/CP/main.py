import argparse
import numpy as np
import random
import pickle
import os
import json

from torch.utils.data import DataLoader
from transformers import BertConfig
from model import MidiBert
from trainer import BERTTrainer
from midi_dataset import MidiDataset


def get_args():
    parser = argparse.ArgumentParser(description='')

    ### path setup ###
    parser.add_argument('--dict_file', type=str, default='../../dict/CP.pkl')
    parser.add_argument('--name', type=str, default='MidiBert')

    ### pre-train dataset ###
    parser.add_argument("--dataset", type=str, nargs='+', default=['pop909','composer', 'pop17k', 'ASAP', 'emopia'])
    
    ### parameter setting ###
    parser.add_argument('--num_workers', type=int, default=5)
    parser.add_argument('--batch_size', type=int, default=12)
    parser.add_argument('--mask_percent', type=float, default=0.15, help="Up to `valid_seq_len * target_max_percent` tokens will be masked out for prediction")
    parser.add_argument('--max_seq_len', type=int, default=512, help='all sequences are padded to `max_seq_len`')
    parser.add_argument('--hs', type=int, default=768)
    parser.add_argument('--epochs', type=int, default=500, help='number of training epochs')
    parser.add_argument('--lr', type=float, default=2e-5, help='initial learning rate')
    
    ### cuda ###
    parser.add_argument("--cpu", action="store_true")   # default: False
    parser.add_argument("--cuda_devices", type=int, nargs='+', default=[0,1,2,3], help="CUDA device ids")

    args = parser.parse_args()

    return args


def load_data(dataset):
    to_concat = []
    
    if 'pop909' in dataset:
        #root = '../prepare_data/CP/CP_data/pop909_'
        root = '/home/yh1488/NAS-189/home/CP_data/pop909_'
        X_train = np.load(root+'train.npy', allow_pickle=True)
        X_val = np.load(root+'valid.npy', allow_pickle=True)
        X_test = np.load(root+'test.npy', allow_pickle=True)
        POP909 = np.concatenate((X_train, X_val, X_test), axis=0)
        print('   POP909:', POP909.shape)
        to_concat.append(POP909)

    if 'composer' in dataset:
        #composer_root = '../prepare_data/CP/CP_data/composer_cp_'
        composer_root = '/home/yh1488/NAS-189/homes/wazenmai/datasets/MIDI-BERT/composer_dataset/CP/composer_cp_'
        composer_train = np.load(composer_root+'train.npy', allow_pickle=True)
        composer_valid = np.load(composer_root+'valid.npy', allow_pickle=True)
        composer_test = np.load(composer_root+'test.npy', allow_pickle=True)
        composer = np.concatenate((composer_train, composer_valid, composer_test), axis=0)
        print('   Composer:', composer.shape)
        to_concat.append(composer)

    if 'pop17k' in dataset:
        #pop17k_path = '../prepare_data/CP/CP_data/pop17k.npy'
        pop17k_path = '/home/yh1488/NAS-189/home/CP_data/ai17k.npy'
        pop17k = np.load(pop17k_path, allow_pickle=True)
        print('   pop17k:', pop17k.shape)
        to_concat.append(pop17k)

    if 'ASAP' in dataset:
        #ASAP_path = '../prepare_data/CP/CP_data/ASAP_CP.npy'
        ASAP_path = '/home/yh1488/NAS-189/homes/wazenmai/MIDI-BERT/for_wazenmai/prepare_CP/ASAP_CP.npy'
        ASAP = np.load(ASAP_path, allow_pickle=True)
        print('   ASAP:', ASAP.shape)
        to_concat.append(ASAP)

    if 'emopia' in dataset:
        #emopia_root = '../prepare_data/CP/CP_data/emopia_'
        emopia_root = '/home/yh1488/NAS-189/homes/wazenmai/datasets/MIDI-BERT/emopia_dataset/CP/emopia_'
        X_train = np.load(emopia_root+'train.npy', allow_pickle=True)
        X_val = np.load(emopia_root+'valid.npy', allow_pickle=True)
        X_test = np.load(emopia_root+'test.npy', allow_pickle=True)
        emopia = np.concatenate((X_train, X_val, X_test), axis=0)
        print('   emopia:', emopia.shape)
        to_concat.append(emopia)
    
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

    print("\nLoading Dataset", args.dataset) 
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
    #save_dir = '/home/yh1488/NAS-189/homes/yh1488/BERT/cp_result/pretrain/'+args.name
    os.makedirs(save_dir, exist_ok=True)
    filename = save_dir + 'model-' + args.name + '.ckpt'
    print("   save model at {}".format(filename))

    best_acc, best_epoch = 0, 0
    bad_cnt = 0

    for epoch in range(args.epochs):
        if bad_cnt >= 30:
            print('valid acc not improving for 30 epochs')
            break
        train_loss, train_acc = trainer.train()
        valid_loss, valid_acc = trainer.valid()

        weighted_score = [x*y for (x,y) in zip(valid_acc, midibert.n_tokens)]
        avg_acc = sum(weighted_score)/sum(midibert.n_tokens)
        
        is_best = avg_acc > best_acc
        best_acc = max(avg_acc, best_acc)
        
        if is_best:
            bad_cnt, best_epoch = 0, epoch
        else:
            bad_cnt += 1
        
        print('epoch: {}/{} | Train Loss: {} | Train acc: {} | Valid Loss: {} | Valid acc: {}'.format(
            epoch+1, args.epochs, train_loss, train_acc, valid_loss, valid_acc))

        trainer.save_checkpoint(epoch, best_acc, valid_acc, 
                                valid_loss, train_loss, is_best, filename)


        with open(os.path.join(save_dir, 'log'), 'a') as outfile:
            outfile.write('Epoch {}: train_loss={}, valid_loss={}, train_acc={}, valid_acc={}\n'.format(
                epoch+1, train_loss, valid_loss, train_acc, valid_acc))


if __name__ == '__main__':
    main()
