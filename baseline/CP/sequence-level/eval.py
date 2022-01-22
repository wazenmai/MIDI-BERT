# -*- coding: utf-8 -*-
"""
Created on Dec 14 2020 
@author: Yi-Hui (Sophia) Chou

Updated on May 10 2021
@author: I-Chun (Bronwin) Chen
"""
import sys
sys.path.append('../../CP')

import torch
import pickle
import argparse
import itertools
import numpy as np
import matplotlib.pyplot as plt

from model import SAN
from pop_dataset import PopDataset
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from sklearn.metrics import confusion_matrix

def get_args():
    parser = argparse.ArgumentParser(description='Argument Parser for sequence-level tasks')
    
    ### mode ###
    parser.add_argument('--task', choices=['composer', 'emotion'], required=True)
    parser.add_argument('--draw', action='store_true', help='draw and save confusion matrix')
    ### path setup ### 
    parser.add_argument('--input', type=str, default='../../../data/CP',help='Path to input numpy folder for composer dataset', required=True)
    parser.add_argument('--dict', type=str, default='../../../BERT/dict/CP.pkl')
    parser.add_argument('--ckpt', type=str, help='Checkpoint folder name', required=True)

    ### parameter setting ###
    parser.add_argument('--cuda', type=int, default=0, help='Specify cuda number')
    parser.add_argument('--class', type=int, help="the class number")

    if args.task == "composer":
        args.num_of_class = 8
    elif args.task == "emotion":
        args.num_of_class = 4

    args = parser.parse_args()
    return args

def save_cm_fig(old_cm, classes, normalize=True, title=None, task=None):
    if normalize:
        old_cm = old_cm.astype('float') * 100 / old_cm.sum(axis=1)[:, None]

    if task == "composer":
        # adjust order
        temp_cm = np.zeros(old_cm.shape)
        for i in range(8):
            temp_cm[0][i] = old_cm[1][i] # Clayderman
            temp_cm[1][i] = old_cm[7][i] # Yiruma
            temp_cm[2][i] = old_cm[3][i] # Hancock
            temp_cm[3][i] = old_cm[2][i] # Einaudi
            temp_cm[4][i] = old_cm[5][i] # Hisaishi
            temp_cm[5][i] = old_cm[6][i] # Ryuchi
            temp_cm[6][i] = old_cm[0][i] # Bethel
            temp_cm[7][i] = old_cm[4][i] # Hillsong
        
        cm = np.zeros(temp_cm.shape)
        for i in range(8):
            cm[i][0] = temp_cm[i][1]
            cm[i][1] = temp_cm[i][7]
            cm[i][2] = temp_cm[i][3]
            cm[i][3] = temp_cm[i][2]
            cm[i][4] = temp_cm[i][5]
            cm[i][5] = temp_cm[i][6]
            cm[i][6] = temp_cm[i][0]
            cm[i][7] = temp_cm[i][4]

    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    if title != None:
        plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, fontsize=20)
    plt.yticks(tick_marks, classes, fontsize=20)

    fmt = '.2f' if normalize else 'd'
    print(fmt)
    threshold = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                horizontalalignment='center',
                fontsize=12,
                color='white' if cm[i, j] > threshold else 'black')
    plt.xlabel('predicted')
    plt.ylabel('true')
    plt.tight_layout()

    plt.savefig('cm.jpg')

def main():
    args = get_args()
    cuda_num = args.cuda
    cuda_str = 'cuda:'+str(cuda_num)
    device = torch.device(cuda_str if torch.cuda.is_available() else 'cpu')

    # load dict
    print("loading dictionary...")
    dict_path = args.dict
    with open(dict_path, 'rb') as f:
        e2w, w2e = pickle.load(f)

    # Load model
    print('Loading model...')
    num_of_class = args.num_of_class
    model = SAN(num_of_dim=num_of_class, e2w=e2w, vocab_size=len(e2w), embedding_size=768, r=4)
    best_ckpt = './experiments/' + args.ckpt + '/SAN_' + args.task + '.pth'
    model.load_state_dict(torch.load(best_ckpt, map_location='cpu')) # already is ['state_dict']
    model = model.to(device)
    model.eval()

    # Load testing data
    inputs = args.input
    if args.task == "composer":
        X_test = torch.tensor(np.load(inputs + "/composer_test.npy"))
        y_test = torch.tensor(np.load(inputs + "/composer_test_ans.npy"))
    elif args.task == "emotion":
        X_test = torch.tensor(np.load(inputs + "/emopia_test.npy"))
        y_test = torch.tensor(np.load(inputs + "/emopia_test_ans.npy"))
    length = len(X_test)
    _X, _y = X_test.to(device, dtype=torch.long), y_test.to(device, dtype=torch.long)

    # Predicted result
    print('Predicting...')
    valid_acc, cnt_sum = 0, 0
    CM = np.zeros((num_of_class, num_of_class))
    with torch.no_grad():
        for ind in range(0, length, 300):
            x, y = _X[ind:ind+300], _y[ind:ind+300]
            x, y = x.to(device).long(), y.to(device).long()   # x: (batch, seq, class), y: (batch, seq)
            
            y_hat = model(x) 
            output = np.argmax(y_hat.cpu().detach().numpy(), axis=-1)
            output = output.astype(float)
            output = torch.from_numpy(output).to(device).long()

            acc = torch.sum(output == y).float()
            valid_acc += acc

            y = y.cpu()
            output = output.cpu()
            cm = confusion_matrix(y, output, labels=[i for i in range(num_of_class)])
            CM = np.add(CM, cm)
        
        acc, sum = 0, 0
        for i in range(0, num_of_class):
            acc += CM[i][i]
            for j in range(0, num_of_class):
                sum += CM[i][j]
        
        print('confusion matrix:\n', CM)
        print('overall acc:', (valid_acc/y.shape[0]).item())
        
    if args.draw and args.task == "composer":
        save_cm_fig(CM, ['C', 'Y', 'H', 'E', 'J', 'S', 'M', 'W'], True, "Composer Classification", args.task)
    if args.draw and args.task == "emotion":
        save_cm_fig(CM, ['HAHV', 'HALV', 'LALV', 'LAHV'], True, "Emotion Classification", args.task)

if __name__ == '__main__':
    main()
