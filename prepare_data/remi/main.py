from model import *
import numpy as np
import argparse
import pathlib
import glob

def get_args():
    parser = argparse.ArgumentParser(description='')
    ### mode ###
    parser.add_argument('-t', '--task', choices=['melody', 'velocity', 'composer', 'emotion'])

    ### path ###
    parser.add_argument('--dict', default='../../dict/remi.pkl')
    parser.add_argument('--dataset', choices=["pop909", "pop17k", "ASAP", "pianist8", "emopia"], required=True)

    ### parameter ###
    parser.add_argument('--max_len', default=512)
    
    ### output ###    
    parser.add_argument('--dir', default="../../data/remi")

    args = parser.parse_args()

    if args.task == 'melody' and args.dataset != 'pop909':
        print('[error] melody task is only supported for pop909 dataset')
        exit(1)
    elif args.task == 'composer' and args.dataset != 'pianist8':
        print('[error] composer task is only supported for pianist8 dataset')
        exit(1)
    elif args.task == 'emotion' and args.dataset != 'emopia':
        print('[error] emotion task is only supported for emopia dataset')
        exit(1)
    return args


def extract(files, args, model, mode):
    print('number of {} files: {}'.format(mode, len(files)))  
    segments, ans = model.prepare_data(files, args.task, int(args.max_len))
    print('segment shape', segments.shape)
    if args.dataset == 'pianist8':
        output_file = args.dir + '/composer_remi_' + mode + '.npy'
    elif args.dataset == 'emopia':
        output_file = args.dir + '/' + args.dataset + '_remi_' + mode + '.npy'
    else:
        output_file = args.dir + '/' + args.dataset + '_' + mode + '.npy'

    np.save(output_file, segments)

    if args.task != None:
        print('ans shape', ans.shape)
        if args.task == 'melody' or args.task == 'velocity':
            ans_file = args.dir + '/' + args.dataset + '_' + mode + '_' + args.task[:3] + 'ans.npy'   
        elif args.task == 'composer' or args.task == 'emotion':
            ans_file = args.dir + '/' + args.dataset + '_cp_' + mode + '_ans.npy'   
        np.save(ans_file, ans)


def main(): 
    args = get_args()
    pathlib.Path(args.dir).mkdir(parents=True, exist_ok=True)
    
    # initialize model
    model = REMI(dict=args.dict)

    if args.dataset == 'pop909':
        files = glob.glob('../../Dataset/pop909_aligned/*.mid')  
    elif args.dataset == 'pop1k7':
        files = glob.glob('../../Dataset/pop1k7/*/*.mid')
    elif args.dataset == 'ASAP':
        files = pickle.load(open('../../Dataset/ASAP_song.pkl'))
        for i, f in enumerate(files):
            files[i] = '../../Dataset/ASAP/' + f
    elif args.dataset == 'pianist8':
        train_files = glob.glob('../../Dataset/pianist8/train/*.mid')
        valid_files = glob.glob('../../Dataset/pianist8/valid/*.mid')
        test_files = glob.glob('../../Dataset/pianist8/test/*.mid')
    elif args.dataset == 'emopia':
        train_files = glob.glob('../../Dataset/emopia/train/*.mid')
        valid_files = glob.glob('../../Dataset/emopia/valid/*.mid')
        test_files = glob.glob('../../Dataset/emopia/test/*.mid')
    else:
        print('not supported')
        exit(1)

    print('number of files', len(files), '\n')

    if args.task:
        # split to 8:1:1 for train, valid, test set
        test_len = (-1) * int(len(files)*0.1)
        train_files = files[: 2*test_len]
        valid_files = files[2*test_len : test_len]
        test_files = files[test_len :]
        
        extract(train_files, args, model, 'train')
        extract(valid_files, args, model, 'valid')
        extract(test_files, args, model, 'test')
    elif args.task == 'composer' or args.task == 'emotion':
        extract(train_files, args, model, 'train')
        extract(valid_files, args, model, 'valid')
        extract(test_files, args, model, 'test')
    else:
        # in one single file
        extract(files, args, model, 'all')


if __name__ == '__main__':
    main()
