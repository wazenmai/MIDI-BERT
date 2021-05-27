import numpy as np
from model_lstm import LSTM_Net
from model_finetune import LSTM_Finetune
import torch
from sklearn.metrics import confusion_matrix
from cm_fig import save_cm_fig
import os
import argparse


def get_args():
    parser = argparse.ArgumentParser(description='Argument Parser for downstream evaluation')
    ### mode ###
    parser.add_argument('-t', '--task', choices=['melody', 'velocity'], required=True)
    parser.add_argument('--finetune', action="store_true")  # default: false

    ### file ###
    parser.add_argument('-i', '--input', type=str)
    parser.add_argument('-a', '--answer', type=str)
    parser.add_argument('-k', '--ckpt', type=str, required=True)

    ### parameter ###
    parser.add_argument('-c', '--cuda', default=0, type=int)
    parser.add_argument('-n', '--class_num', type=int)
    parser.add_argument('--layer', type=str, default='12')

    args = parser.parse_args()

    if args.task == 'melody':
        args.answer = '/home/yh1488/NAS-189/home/remi_data/POP909remi_melans.npy'
        args.class_num = 3
    elif args.task == 'velocity':
        args.answer = '/home/yh1488/NAS-189/home/remi_data/POP909remi_velans.npy'
        args.class_num = 6

    if args.finetune:
        args.input='/home/yh1488/NAS-189/home/BERT/remi_embed/POPtest_'+args.layer+'.npy'
    else:
        args.input='/home/yh1488/NAS-189/home/remi_data/POP909remi.npy'
    return args


def load_data(input_file, ans_file, finetune):
    # prepare data to 80:10:10
    all_data = np.load(input_file)
    all_ans = np.load(ans_file)
    _, _, y_test = np.split(all_ans, [int(.8 * len(all_ans)), int(.9 * len(all_ans))])
    
    
    if finetune:
        return all_data, y_test
    else:
        _, _, X_test = np.split(all_data, [int(.8 * len(all_data)), int(.9 * len(all_data))])
        return X_test, y_test


def main():
    args = get_args()
    cuda_num = args.cuda 
    cuda_str = "cuda:"+str(cuda_num)
    device = torch.device(cuda_str if torch.cuda.is_available() else 'cpu')

    # Load model
    best_mdl = args.ckpt
    mode = 'finetune' if args.finetune else 'LSTM'
    print('Loading model from [{}/{}]...'.format(mode, best_mdl.split('/')[-2]))
    if args.finetune:
        model = LSTM_Finetune(label_class=args.class_num)
    else:
        model = LSTM_Net(label_class=args.class_num)
    model.load_state_dict(torch.load(best_mdl, map_location='cpu'))    # already is ['state_dict']
    model = model.to(device)
    model.eval()

    # Load testing data
    print('Loading testing data...')
    _X, _y =  load_data(args.input, args.answer, args.finetune)
    print('length', len(_X))
    length = len(_X)
    _X, _y = torch.tensor(_X), torch.tensor(_y)

    if args.finetune:
        _X, _y = _X.to(device, dtype=torch.float), _y.to(device, dtype=torch.float)
    else:
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
        
#            print('accuracy:', acc/sum)
    
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

        bORf = 'finetune' if args.finetune else 'baseline'
        _title = 'remi: ' + args.task + ' task (' + bORf + ')'
        save_cm_fig(CM, classes=target_names, normalize=True, title=_title)


if __name__ == '__main__':
    main()
