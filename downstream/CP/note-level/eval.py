import numpy as np
from model import LSTM_Net
import torch
import pickle
import argparse
from sklearn.metrics import confusion_matrix
from cm_fig import save_cm_fig

def get_args():
    parser = argparse.ArgumentParser(description='Argument Parser for downstream evaluation')
    ### mode ###
    parser.add_argument('--task', choices=['melody', 'velocity'], required=True)

    ### file ###
    parser.add_argument('--dict', default = '../../dict/CP.pkl', type=str)
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
    
    root = '../../data/CP/pop909_'
    X_test = np.load(root+'test.npy', allow_pickle=True)
    y_test = np.load(root+'test_'+task[:3]+'ans.npy', allow_pickle=True)
    
    return X_test, y_test



def main():
    args = get_args()
    cuda_str = "cuda:"+str(args.cuda)
    device = torch.device(cuda_str if torch.cuda.is_available() else 'cpu')

    # load dict
    print("Loading dictionary...")
    with open(args.dict, 'rb') as f:
        e2w, w2e = pickle.load(f)

    # Load model
    best_mdl = args.ckpt
    print('Loading model from [{}/{}]...'.format('LSTM', best_mdl.split('/')[-2]))
    model = LSTM_Net(e2w=e2w, class_num=args.class_num)
    model.load_state_dict(torch.load(best_mdl, map_location='cpu'))    # already is ['state_dict']
    model = model.to(device)
    model.eval()

    # Load testing data
    print('Loading testing data...')
    _X, _y = load_data(args.task)
    _X, _y = torch.tensor(_X), torch.tensor(_y)
    _X, _y = _X.to(device, dtype=torch.long), _y.to(device, dtype=torch.float)
    
    # Predicted result
    print('Predicting...')
    with torch.no_grad():
        y_hat = model(_X) 
        output = np.argmax(y_hat.cpu().detach().numpy(), axis=-1)
        output = output.astype(float)
        output = torch.from_numpy(output).to(device).long()

        attn = (_y != 0).float()      # != bar pad word

        acc = torch.sum((output == _y).float() * attn)
        acc /= torch.sum(attn)
        print('accuracy:', acc.item())

        if args.task == 'melody':
            target_names = ['M', 'B', 'A']
        elif args.task == 'velocity':
            target_names = ['pp','p','mp','mf','f','ff']
   
        output = output.detach().cpu().numpy()
        _y = _y.detach().cpu().numpy().astype(int)
        output = output.reshape(-1,1)
        _y = _y.reshape(-1,1)

        cm = confusion_matrix(_y, output)
        print(cm)
        
        _title = 'LSTM (CP): ' + args.task + ' task'
        save_cm_fig(cm, classes=target_names, normalize=True, title=_title)
    

if __name__ == '__main__':
    main()
