import sys
sys.path.append('../../CP')

import torch
import pickle
import argparse
import itertools
import numpy as np
import matplotlib.pyplot as plt

from model import LSTM_Net, SAN
from pop_dataset import PopDataset
from torch.utils.data import DataLoader
from sklearn.metrics import confusion_matrix

def get_args():
    parser = argparse.ArgumentParser(description='Argument Parser for sequence-level baseline')

    ### mode ###
    parser.add_argument('--task', choices=['composer', 'emotion'], required=True)
    parser.add_argument('--draw', action='store_true', help='draw and save confusion matrix')
    ### path setup ### 
    parser.add_argument('--input', type=str, default='../../../data/remi',help='Path to input numpy folder for composer dataset', required=True)
    parser.add_argument('--dict', type=str, default='../../../BERT/dict/remi.pkl')
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

def load_data(input_file, ans_file):
    All, Ans = np.load(input_file), np.load(ans_file)
    _, _, X_test = np.split(All, [int(.7*len(All)),int(.85*len(All))])
    _, _, y_test = np.split(Ans, [int(.7*len(Ans)),int(.85*len(Ans))])
    return X_test, y_test

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
    cuda_str = "cuda:" + str(cuda_num)
    device = torch.device(cuda_str if torch.cuda.is_available() else 'cpu')

    print('Loading dictionary...')
    dict_path = args.dict
    e2w, w2e = pickle.load(open(dict_path, 'rb'))

    print('Loading model...')
    num_of_class = args.num_of_class
    model = SAN(num_of_dim=num_of_class, vocab_size=len(e2w), embedding_size=768, r=4)
    best_ckpt = './experiments/' + args.ckpt + '/SAN_' + args.task + '.pth'
    model.load_state_dict(torch.load(best_ckpt, map_location='cpu')) # already is ['state_dict']
    model = model.to(device)
    model.eval()

    # Load testing data
    print('Loading testing data...')
    inputs = args.input
    if args.task == "composer":
        X_test = torch.tensor(np.load(inputs + "/composer_test.npy"))
        y_test = torch.tensor(np.load(inputs + "/composer_test_ans.npy"))
    elif args.task == "emotion":
        X_test = torch.tensor(np.load(inputs + "/emopia_test.npy"))
        y_test = torch.tensor(np.load(inputs + "/emopia_test_ans.npy"))
    testset = PopDataset(X=X_test, y=y_test)
    test_loader = DataLoader(testset, batch_size = 1, shuffle = True, drop_last = True)
    
    # Predicted result
    print('Predicting...')
    CM = np.zeros((num_of_clas, num_of_class))
    with torch.no_grad():
        for X, y in test_loader:
            X, y = X.to(device, dtype=torch.long), y.to(device, dtype=torch.float)
            y_hat = model(X)
            y_hat = y_hat.reshape((-1, num_of_class))
            y = y.reshape((-1)).to(dtype = torch.long)
            _, pred_label = torch.max(y_hat, dim=1)

            y_np = y.detach().cpu().numpy()
            pred_np = pred_label.detach().cpu().numpy()

            cm = confusion_matrix(y_np, pred_np, labels=[i for i in range(num_of_class)])
            CM = np.add(CM, cm)
        
        acc, sum = 0, 0
        for i in range(0, num_of_class):
            acc += CM[i][i]
            for j in range(0, num_of_class):
                sum += CM[i][j]
        
        print('confusion matrix:\n', CM)
        print('accuracy:', acc/sum)

    if args.draw and args.task == "composer":
        save_cm_fig(CM, ['C', 'Y', 'H', 'E', 'J', 'S', 'M', 'W'], True, "Composer Classification", args.task)
    if args.draw and args.task == "emotion":
        save_cm_fig(CM, ['HAHV', 'HALV', 'LALV', 'LAHV'], True, "Emotion Classification", args.task)

if __name__ == '__main__':
    main()
