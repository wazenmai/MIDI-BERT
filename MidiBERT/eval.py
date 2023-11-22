"""
Evaluate the model on fine-tuning task (melody, velocity, composer, emotion)
Return loss, accuracy, confusion matrix.
"""
import argparse
import numpy as np
import random
import pickle
import os
import copy
import shutil
import json
import tqdm
from sklearn.metrics import confusion_matrix
from cm_fig import save_cm_fig

from torch.utils.data import DataLoader
import torch
import torch.nn as nn
from transformers import BertConfig

from MidiBERT.model import MidiBert_CP, MidiBert_remi
from MidiBERT.finetune_trainer import FinetuneTrainer
from MidiBERT.finetune_dataset import FinetuneDataset
from MidiBERT.finetune_model import TokenClassification, SequenceClassification


def get_args():
    parser = argparse.ArgumentParser(description='')
    ### representation ###
    parser.add_argument('--repr', type=str, choices=['CP', 'remi'], required=True) 

    ### mode ###
    parser.add_argument('--task', choices=['melody', 'velocity','composer', 'emotion'], required=True)
    
    ### path setup ###
    parser.add_argument('--dict_dir', type=str, default='data_creation/prepare_data/dict')
    parser.add_argument('--ckpt', type=str, default='')

    ### parameter setting ###
    parser.add_argument('--num_workers', type=int, default=5)
    parser.add_argument('--class_num', type=int)
    parser.add_argument('--batch_size', type=int, default=12)
    parser.add_argument('--max_seq_len', type=int, default=512, help='all sequences are padded to `max_seq_len`')
    parser.add_argument('--hs', type=int, default=768)
    parser.add_argument("--index_layer", type=int, default=12, help="number of layers")
    parser.add_argument('--lr', type=float, default=2e-5, help='initial learning rate')
    
    ### cuda ###
    parser.add_argument('--cpu', action="store_true")  # default: false
    parser.add_argument("--cuda_devices", type=int, nargs='+', default=[0,1], help="CUDA device ids")

    args = parser.parse_args()

    root = f'MidiBERT/result/finetune_{args.repr}/'

    if args.task == 'melody':
        args.class_num = 4
        args.ckpt = root + 'melody_default/model_best.ckpt' if args.ckpt=='' else args.ckpt
    elif args.task == 'velocity':
        args.class_num = 7
        args.ckpt = root + 'velocity_default/model_best.ckpt' if args.ckpt=='' else args.ckpt
    elif args.task == 'composer':
        args.class_num = 8
        args.ckpt = root + 'composer_default/model_best.ckpt' if args.ckpt=='' else args.ckpt
    elif args.task == 'emotion':
        args.class_num = 4
        args.ckpt = root + 'emotion_default/model_best.ckpt' if args.ckpt=='' else args.ckpt

    return args


def load_data(dataset, task, rep):
    data_root = f'Data/{rep}_data'

    if dataset == 'emotion':
        dataset = 'emopia'

    if dataset not in ['pop909', 'composer', 'emopia']:
        print('dataset {} not supported'.format(dataset))
        exit(1)
        
    X_test = np.load(os.path.join(data_root, f'{dataset}_test.npy'), allow_pickle=True)

    if dataset == 'pop909':
        y_test = np.load(os.path.join(data_root, f'{dataset}_test_{task[:3]}ans.npy'), allow_pickle=True)
    else:
        y_test = np.load(os.path.join(data_root, f'{dataset}_test_ans.npy'), allow_pickle=True)


    print('X_test: {}'.format(X_test.shape))
    print('y_test: {}'.format(y_test.shape))

    return X_test, y_test


def conf_mat(_y, output, task, outdir, rep):
    if task == 'melody':
        target_names = ['M','B','A']
        seq = False
    elif task == 'velocity':
        target_names = ['pp','p','mp','mf','f','ff']
        seq = False
    elif task == 'composer':
        target_names = ['C','Y','H','E','J','S','M','W']
        seq = True
    elif task == 'emotion':
        target_names = ['HAHV', 'HALV', 'LALV', 'LAHV']
        seq = True
        
    output = output.detach().cpu().numpy()
    output = output.reshape(-1,1)
    _y = _y.reshape(-1,1)
    
    cm = confusion_matrix(_y, output) 
    
    _title = f'BERT (rep): {task} task'
    
    save_cm_fig(cm, classes=target_names, normalize=True,
                title=_title, outdir=outdir, seq=seq)

    return

def evaluate(X_test, y_test, model, device, seq, batch_size, num_workers, max_seq_len=512, index_layer=-1):
    # load data
    testset = FinetuneDataset(X=X_test, y=y_test) 
    test_loader = DataLoader(testset, batch_size=batch_size, num_workers=num_workers)
    print("   len of test_loader",len(test_loader))

    # model
    model.eval()
    pbar = tqdm.tqdm(test_loader, disable=False)
    total_acc, total_cnt = 0, 0
    all_output = torch.empty(y_test.shape)
    ind = 0

    for x, y in pbar:  # (batch, 512, 768)
        batch = x.shape[0]
        x, y = x.to(device), y.to(device)     # x: (batch, 512, _), y_seq: (batch), y_note: (batch, 512)

        # avoid attend to pad word
        if not seq:
            attn = (y != 0).float().to(device)   # (batch, 512)
        else:   
            attn = torch.ones((batch, max_seq_len)).to(device)     # attend each of them

        y_hat = model.forward(x, attn, index_layer)     # seq: (batch, class_num) / token: (batch, 512, class_num)

        # get the most likely choice with max
        output = np.argmax(y_hat.cpu().detach().numpy(), axis=-1)
        output = torch.from_numpy(output).to(device)
        all_output[ind : ind+batch] = output
        ind += batch

        # accuracy
        if not seq:
            acc = torch.sum((y == output).float() * attn)
            total_acc += acc
            total_cnt += torch.sum(attn).item()
        else:
            acc = torch.sum((y == output).float())
            total_acc += acc
            total_cnt += y.shape[0]

    return round(total_acc.item()/total_cnt,4), all_output


def main():
    args = get_args()
    rep = args.repr
    print('-'*50, rep, '-'*50)

    print("Loading Dictionary")
    with open(f'{args.dict_dir}/{rep}.pkl', 'rb') as f:
        e2w, w2e = pickle.load(f)

    print("\nBuilding BERT model")
    configuration = BertConfig(max_position_embeddings=args.max_seq_len,
                                position_embedding_type='relative_key_query',
                                hidden_size=args.hs)

    if args.repr == 'CP':
        midibert = MidiBert_CP(bertConfig=configuration, e2w=e2w, w2e=w2e)
    elif args.repr == 'remi':
        midibert = MidiBert_remi(bertConfig=configuration, e2w=e2w, w2e=w2e)
    
    print("\nLoading Dataset") 
    if args.task == 'melody' or args.task == 'velocity':
        dataset = 'pop909'
        model = TokenClassification(midibert, args.class_num, args.hs)
        seq_class = False
    elif args.task == 'composer' or args.task == 'emotion':
        dataset = args.task
        model = SequenceClassification(midibert, args.class_num, args.hs)
        seq_class = True
    else:
        raise NotImplementedError(f"not defined task {args.task}")
        
    X_test, y_test = load_data(dataset, args.task, rep)
    
    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else 'cpu')

    print('\nLoad a finetuned model from', args.ckpt)  
    best_mdl = args.ckpt 
    checkpoint = torch.load(best_mdl, map_location='cpu')
    if "module" in list(checkpoint["state_dict"].keys())[0]:
        # remove module
        from collections import OrderedDict
        new_state_dict = OrderedDict()
        for k, v in checkpoint['state_dict'].items():
            name = k[7:]
            new_state_dict[name] = v
        model.load_state_dict(new_state_dict)
    else:
        model.load_state_dict(checkpoint['state_dict'])
    model = model.to(device)


    index_layer = int(args.index_layer)-13
    #trainer = FinetuneTrainer(midibert, train_loader, valid_loader, test_loader, index_layer, args.lr, args.class_num,
    #                            args.hs, y_test.shape, args.cpu, args.cuda_devices, model, seq_class)
    
  
    test_acc, all_output = evaluate(X_test, y_test, model, device, seq_class, args.batch_size, args.num_workers, args.max_seq_len, index_layer)
    print('test_acc: {}'.format(test_acc))

    outdir = os.path.dirname(args.ckpt)
    conf_mat(y_test, all_output, args.task, outdir, rep)

if __name__ == '__main__':
    main()
