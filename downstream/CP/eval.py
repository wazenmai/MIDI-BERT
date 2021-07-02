import numpy as np
from model_lstm import LSTM_Net
from model_finetune import LSTM_Finetune
import torch
import pickle
import argparse
from sklearn.metrics import confusion_matrix
from cm_fig import save_cm_fig

def get_args():
    parser = argparse.ArgumentParser(description='Argument Parser for downstream evaluation')
    ### mode ###
    parser.add_argument('-t', '--task', choices=['melody', 'velocity'], required=True)
    parser.add_argument('--finetune', action="store_true")  # default: false

    ### file ###
    parser.add_argument('-i', '--input', type=str)
    parser.add_argument('-a', '--answer', type=str)
    parser.add_argument('-d', '--dict', default = 'CP.pkl', type=str)
    parser.add_argument('-k', '--ckpt', type=str, required=True)

    ### parameter ###
    parser.add_argument('--hs', type=int, default=256)
    parser.add_argument('--layer', type=str, default='12', help='specify embedding layer index')
    parser.add_argument('-c', '--cuda', default=0, type=int)
    parser.add_argument('-n', '--class_num', type=int)

    args = parser.parse_args()

    if args.task == 'melody':
        args.answer = '/home/yh1488/NAS-189/home/CP_data/POP909cp_melans.npy'
        args.class_num = 3
    elif args.task == 'velocity':
        args.answer = '/home/yh1488/NAS-189/home/CP_data/POP909cp_velans.npy'
        args.class_num = 6
    
    if args.finetune:
        #X_train = np.load('/home/yh1488/NAS-189/home/BERT/cp_embed/final/pop909-layer12-train.npy')
        args.input = '/home/yh1488/NAS-189/home/BERT/cp_embed/final/pop909-layer12-test.npy' #+args.layer+'.npy'
    else:
        args.input = '/home/yh1488/NAS-189/home/CP_data/POP909cp.npy'


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

    # load dict
    print("Loading dictionary...")
    with open(args.dict, 'rb') as f:
        e2w, w2e = pickle.load(f)

    # Load model
    best_mdl = args.ckpt
    mode = "finetune" if args.finetune else "LSTM"
    print('Loading model from [{}/{}]...'.format(mode, best_mdl.split('/')[-2]))
    if args.finetune:
        model = LSTM_Finetune(class_num=args.class_num, hidden_size=args.hs)
    else:
        model = LSTM_Net(e2w=e2w, class_num=args.class_num)
    model.load_state_dict(torch.load(best_mdl, map_location='cpu'))    # already is ['state_dict']
    model = model.to(device)
    model.eval()

    # Load testing data
    print('Loading testing data...')
    _X, _y =  load_data(args.input, args.answer, args.finetune)
    print('length', len(_X))
    _X, _y = torch.tensor(_X), torch.tensor(_y)

    if args.finetune:
        _X, _y = _X.to(device, dtype=torch.float), _y.to(device, dtype=torch.float)
    else:
        _X, _y = _X.to(device, dtype=torch.long), _y.to(device, dtype=torch.float)
    
    # Predicted result
    print('Predicting...')
    valid_acc = 0
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
            target_names = ['melody', 'bridge', 'piano']
        elif args.task == 'velocity':
            target_names = ['pp','p','mp','mf','f','ff']
   
        output = output.detach().cpu().numpy()
        _y = _y.detach().cpu().numpy().astype(int)
        output = output.reshape(-1,1)
        _y = _y.reshape(-1,1)

        cm = confusion_matrix(_y, output)
        print(cm)
        bORf = 'finetune' if args.finetune else 'baseline'
        _title = 'CP: ' + args.task + ' task (' + bORf + ')'
        save_cm_fig(cm, classes=target_names, normalize=True,
                    title=_title)
        
if __name__ == '__main__':
    main()
