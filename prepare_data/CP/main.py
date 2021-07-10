from model import *
import numpy as np
import argparse
import os
import pathlib
import glob


def get_args():
    parser = argparse.ArgumentParser(description='')
    ### mode ###
    parser.add_argument('-t', '--task', choices=['melody', 'velocity'])

    ### path ###
    parser.add_argument('--dict', default='../../dict/CP.pkl')
    parser.add_argument('--dataset', choices=["pop909", "pop1k7"], required=True)

    ### parameter ###
    parser.add_argument('--max_len', default=512)
    
    ### output ###    
    parser.add_argument('--dir', default="../../data/CP")

    args = parser.parse_args()

    if args.task == 'melody' and args.dataset != 'pop909':
        print('[error] melody task is only supported for pop909 dataset')
        exit(1)
    return args


def extract(files, args, model, mode):
    print('number of {} files: {}'.format(mode, len(files)))  
    segments, ans = model.prepare_data(files, args.task, int(args.max_len))
    print('segment shape', segments.shape)
    output_file = args.dir + '/' + args.dataset + '_' + mode + '.npy'

    np.save(output_file, segments)
    if args.task != None:
        print('ans shape', ans.shape)
        ans_file = args.dir + '/' + args.dataset + '_' + mode + '_' + args.task[:3] + 'ans.npy'   
        np.save(ans_file, ans)
    


def main(): 
    args = get_args()
    pathlib.Path(args.dir).mkdir(parents=True, exist_ok=True)
    
    # initialize model
    model = CP(dict=args.dict)

    root = '/home/yh1488/NAS-189/home/'
    if args.dataset == 'pop909':
        files = glob.glob(root+'Dataset/pop909_aligned/*.mid')  
    elif args.dataset == 'pop1k7':
        files = glob.glob(root+'Dataset/pop1k7/*/*.mid')
    else:
        print('not supported')
        exit(1)

    print('number of files', len(files))

    if args.task:
        # split to 8:1:1 for train, valid, test set
        test_len = (-1) * int(len(files)*0.1)
        train_files = files[: 2*test_len]
        valid_files = files[2*test_len : test_len]
        test_files = files[test_len :]
        
        extract(train_files, args, model, 'train')
        extract(valid_files, args, model, 'valid')
        extract(test_files, args, model, 'test')
    else:
        # in one single file
        extract(files, args, model, 'all')

if __name__ == '__main__':
    main()
