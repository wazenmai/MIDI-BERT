import os
import torch
import argparse
import numpy as np
from model import LSTM_Net
from cm_fig import save_cm_fig
from sklearn.metrics import confusion_matrix

def get_args():
    parser = argparse.ArgumentParser(description='Argument Parser for downstream evaluation')
    ### mode ###
    parser.add_argument('--task', choices=['melody', 'velocity'], required=True)

    ### file ###
    parser.add_argument('--ckpt', type=str, required=True)

    ### parameter ###
    parser.add_argument('--cuda', default=0, type=int)
    parser.add_argument('--class_num', type=int)

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
    X_test = np.load(root+'test.npy', allow_pickle=True)
    y_test = np.load(root+'test_'+task[:3]+'ans.npy', allow_pickle=True)
    
    return X_test, y_test


def main():
    args = get_args()
    cuda_num = args.cuda 
    cuda_str = "cuda:"+str(cuda_num)
    device = torch.device(cuda_str if torch.cuda.is_available() else 'cpu')

    # Load model
    best_mdl = args.ckpt
    print('Loading model from [{}/{}]...'.format('LSTM', best_mdl.split('/')[-2]))
    model = LSTM_Net(class_num=args.class_num)
    model.load_state_dict(torch.load(best_mdl, map_location='cpu'))    # already is ['state_dict']
    model = model.to(device)
    model.eval()

    # Load testing data
    print('Loading testing data...')
    _X, _y =  load_data(args.task)
    length = len(_X)
    _X, _y = torch.tensor(_X), torch.tensor(_y)

    _X, _y = _X.to(device, dtype=torch.long), _y.to(device, dtype=torch.float)
    
    # Predicted result
    print('Predicting...')
    num = args.class_num + 1
    CM = np.zeros((num, num))
    with torch.no_grad():
        for ind in range(0, length, 300):
            X, y = _X[ind:ind+300], _y[ind:ind+300]
            X, y = X.to(device, dtype=torch.long), y.to(device, dtype=torch.float)
            y_hat = model(X)
            y_hat = y_hat.reshape((-1,num))
            y_hat = y_hat.squeeze()
            y = y.reshape((-1)).to(dtype = torch.long)
            _, pred_label = torch.max(y_hat, dim=1)

            y_np = y.detach().cpu().numpy()
            pred_np = pred_label.detach().cpu().numpy()

            # confusion matrix
            cm = confusion_matrix(y_np, pred_np, labels=[i for i in range(num)])
            CM = np.add(CM, cm)
            acc, sum = 0, 0
            for i in range(1,num):
                acc += cm[i][i]
                for j in range(1,num):
                    sum += cm[i][j]
        
    
        acc, sum = 0, 0
        for i in range(1,num):
            acc += CM[i][i]
            for j in range(1,num):
                sum += CM[i][j]
        
        print('cm:\n', CM)
        print('accuracy:', acc/sum)

        if args.task == 'melody':
            target_names = ['melody', 'bridge', 'piano']
        elif args.task == 'velocity':
            target_names = ['pp','p','mp','mf','f','ff']

        _title = 'remi: ' + args.task + ' task (baseline)'
        save_cm_fig(CM, classes=target_names, normalize=True, title=_title)


if __name__ == '__main__':
    main()
