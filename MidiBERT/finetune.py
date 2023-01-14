import argparse
import numpy as np
import pickle
import os
import random

from torch.utils.data import DataLoader
import torch
from transformers import BertConfig

from MidiBERT.model import MidiBert_CP, MidiBert_remi
from MidiBERT.finetune_trainer import FinetuneTrainer
from MidiBERT.finetune_dataset import FinetuneDataset

from matplotlib import pyplot as plt

def get_args():
    parser = argparse.ArgumentParser(description='')

    ### representation ###
    parser.add_argument('--repr', type=str, choices=['CP', 'remi'], required=True) 

    ### mode ###
    parser.add_argument('--task', choices=['melody', 'velocity', 'composer', 'emotion'], required=True)

    ### path setup ###
    parser.add_argument('--dict_dir', type=str, default='data_creation/prepare_data/dict')
    parser.add_argument('--tag', type=str, default='', required=True)
    parser.add_argument('--ckpt', default='')

    ### parameter setting ###
    parser.add_argument('--num_workers', type=int, default=5)
    parser.add_argument('--class_num', type=int)
    parser.add_argument('--batch_size', type=int, default=12)
    parser.add_argument('--max_seq_len', type=int, default=512, help='all sequences are padded to `max_seq_len`')
    parser.add_argument('--hs', type=int, default=768)
    parser.add_argument("--index_layer", type=int, default=12, help="number of layers")
    parser.add_argument('--epochs', type=int, default=10, help='number of training epochs')
    parser.add_argument('--lr', type=float, default=2e-5, help='initial learning rate')
    parser.add_argument('--nopretrain', action="store_true")  # default: false
    
    ### cuda ###
    parser.add_argument("--cpu", action="store_true") # default=False
    parser.add_argument("--cuda_devices", type=int, nargs='+', default=[0,1], help="CUDA device ids")

    args = parser.parse_args()

    if args.task == 'melody':
        args.class_num = 4
    elif args.task == 'velocity':
        args.class_num = 7
    elif args.task == 'composer':
        args.class_num = 8
    elif args.task == 'emotion':
        args.class_num = 4

    if args.ckpt == "":
        args.ckpt = f"MidiBERT/result/pretrain_{args.repr}/test/model_best.ckpt"

    return args


def load_data(dataset, task, rep):
    data_root = f'Data/{rep}_data'

    if dataset == 'emotion':
        dataset = 'emopia'

    if dataset not in ['pop909', 'composer', 'emopia']:
        print(f'Dataset {dataset} not supported')
        exit(1)
        
    X_train = np.load(os.path.join(data_root, f'{dataset}_train.npy'), allow_pickle=True)
    X_val = np.load(os.path.join(data_root, f'{dataset}_valid.npy'), allow_pickle=True)

    if dataset == 'pop909':
        y_train = np.load(os.path.join(data_root, f'{dataset}_train_{task[:3]}ans.npy'), allow_pickle=True)
        y_val = np.load(os.path.join(data_root, f'{dataset}_valid_{task[:3]}ans.npy'), allow_pickle=True)
    else:
        y_train = np.load(os.path.join(data_root, f'{dataset}_train_ans.npy'), allow_pickle=True)
        y_val = np.load(os.path.join(data_root, f'{dataset}_valid_ans.npy'), allow_pickle=True)

    print('X_train: {}, X_valid: {}'.format(X_train.shape, X_val.shape))
    print('y_train: {}, y_valid: {}'.format(y_train.shape, y_val.shape))

    return X_train, X_val, y_train, y_val


def main():
    # set seed
    seed = 2021
    torch.manual_seed(seed)             # cpu
    torch.cuda.manual_seed(seed)        # current gpu
    torch.cuda.manual_seed_all(seed)    # all gpu
    np.random.seed(seed)
    random.seed(seed)

    # argument
    args = get_args()

    print('-'*50, args.repr, '-'*50)

    print("Loading Dictionary")
    with open(f'{args.dict_dir}/{args.repr}.pkl', 'rb') as f:
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
    X_train, X_val, y_train, y_val = load_data(dataset, args.task, args.repr)
    
    trainset = FinetuneDataset(X=X_train, y=y_train)
    validset = FinetuneDataset(X=X_val, y=y_val) 

    train_loader = DataLoader(trainset, batch_size=args.batch_size, num_workers=args.num_workers, shuffle=True)
    print("   len of train_loader",len(train_loader))
    valid_loader = DataLoader(validset, batch_size=args.batch_size, num_workers=args.num_workers)
    print("   len of valid_loader",len(valid_loader))

    print("\nBuilding BERT model")
    configuration = BertConfig(max_position_embeddings=args.max_seq_len,
                                position_embedding_type='relative_key_query',
                                hidden_size=args.hs)

    if args.repr == 'CP':
        midibert = MidiBert_CP(bertConfig=configuration, e2w=e2w, w2e=w2e)
    elif args.repr == 'remi':
        midibert = MidiBert_remi(bertConfig=configuration, e2w=e2w, w2e=w2e)
    
    best_mdl = ''
    if not args.nopretrain:
        best_mdl = args.ckpt
        print("   Loading pre-trained model from", best_mdl.split('/')[-1])
        checkpoint = torch.load(best_mdl, map_location='cpu')
        midibert.load_state_dict(checkpoint['state_dict'])
    
    index_layer = int(args.index_layer)-13
    print("\nCreating Finetune Trainer using index layer", index_layer)
   
    trainer = FinetuneTrainer(midibert, train_loader, valid_loader, index_layer, args.lr, args.class_num,
                                args.hs, args.cpu, args.cuda_devices, None, seq_class)
    
    print("\nTraining Start")
    save_dir = os.path.join(f'MidiBERT/result/finetune_{args.repr}/', args.task + '_' + args.tag)
    os.makedirs(save_dir, exist_ok=True)
    filename = os.path.join(save_dir, 'model.ckpt')
    print("   save model at {}".format(filename))

    best_acc, best_epoch = 0, 0
    bad_cnt = 0

#    train_accs, valid_accs = [], []
    with open(os.path.join(save_dir, 'log'), 'a') as outfile:
        outfile.write("Loading pre-trained model from " + best_mdl.split('/')[-1] + '\n')
        for epoch in range(args.epochs):
            train_loss, train_acc = trainer.train()
            valid_loss, valid_acc = trainer.valid()

            is_best = valid_acc >= best_acc
            best_acc = max(valid_acc, best_acc)
            
            if is_best:
                bad_cnt, best_epoch = 0, epoch
            else:
                bad_cnt += 1
            
            print('epoch: {}/{} | Train Loss: {} | Train acc: {} | Valid Loss: {} | Valid acc: {}'.format(
                epoch+1, args.epochs, train_loss, train_acc, valid_loss, valid_acc))

#            train_accs.append(train_acc)
#            valid_accs.append(valid_acc)
            trainer.save_checkpoint(epoch, train_acc, valid_acc, 
                                    valid_loss, train_loss, is_best, filename)


            outfile.write('Epoch {}: train_loss={}, valid_loss={}, train_acc={}, valid_acc={}\n'.format(
                epoch+1, train_loss, valid_loss, train_acc, valid_acc))
        
            if bad_cnt > 3:
                print('valid acc not improving for 3 epochs')
                break

    # draw figure valid_acc & train_acc
    '''plt.figure()
    plt.plot(train_accs)
    plt.plot(valid_accs)
    plt.title(f'{args.task} task accuracy (w/o pre-training)')
    plt.xlabel('epoch')
    plt.ylabel('accuracy')
    plt.legend(['train','valid'], loc='upper left')
    plt.savefig(f'acc_{args.task}_scratch.jpg')'''

if __name__ == '__main__':
    main()
