from model import *
import numpy as np
import argparse
import pathlib
import glob
import os


def get_args():
    parser = argparse.ArgumentParser(description='')
    ### mode ###
    parser.add_argument('-t', '--task', default='', choices=['melody', 'velocity', 'composer', 'emotion'])

    ### path ###
    parser.add_argument('--dict', type=str, default='../../dict/remi.pkl')
    parser.add_argument('--dataset', type=str, choices=["pop909", "pop1k7", "ASAP", "pianist8", "emopia"])
    parser.add_argument('--input_dir', type=str, default='')

    ### parameter ###
    parser.add_argument('--max_len', type=int, default=512)
    
    ### output ###    
    parser.add_argument('--output_dir', default="../../data/remi")
    parser.add_argument('--name', default="")   # will be saved as "{output_dir}/{name}.npy"

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
    elif args.dataset == None and args.input_dir == None:
        print('[error] Please specify the input directory or dataset')
        exit(1)

    return args


def extract(files, args, model, mode=''):
    assert len(files)
    print(f'number of {mode} files: {len(files)}') 

    segments, ans = model.prepare_data(files, args.task, int(args.max_len))

    dataset = args.dataset if args.dataset != 'pianist8' else 'composer'

    if args.input_dir != '':
        if args.name == '':
            args.name = os.path.basename(os.path.normpath(args.input_dir))
        output_file = os.path.join(args.output_dir, f'{args.name}.npy')
    elif dataset == 'composer' or dataset == 'emopia' or dataset == 'pop909':
        output_file = os.path.join(args.output_dir, f'{dataset}_{mode}.npy')
    elif dataset == 'pop1k7' or dataset == 'ASAP':
        output_file = os.path.join(args.output_dir, f'{dataset}.npy')

    np.save(output_file, segments)
    print(f'Data shape: {segments.shape}, saved at {output_file}')

    if args.task != '':
        if args.task == 'melody' or args.task == 'velocity':
            ans_file = os.path.join(args.output_dir, f'{dataset}_{mode}_{args.task[:3]}ans.npy')
        elif args.task == 'composer' or args.task == 'emotion':
            ans_file = os.path.join(args.output_dir, f'{dataset}_{mode}_ans.npy')
        np.save(ans_file, ans)
        print(f'Answer shape: {ans.shape}, saved at {ans_file}')


def main(): 
    args = get_args()
    pathlib.Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    
    # initialize model
    model = REMI(dict=args.dict)

    if args.dataset == 'pop909':
        dataset = args.dataset
    elif args.dataset == 'emopia':
        dataset = 'EMOPIA_1.0'
    elif args.dataset == 'pianist8':
        dataset = 'joann8512-Pianist8-ab9f541'

    if args.dataset == 'pop909' or args.dataset == 'emopia':
        train_files = glob.glob(f'../../Dataset/{dataset}/train/*.mid')
        valid_files = glob.glob(f'../../Dataset/{dataset}/valid/*.mid')
        test_files = glob.glob(f'../../Dataset/{dataset}/test/*.mid')

    elif args.dataset == 'pianist8':
        train_files = glob.glob(f'../../Dataset/{dataset}/train/*/*.mid')
        valid_files = glob.glob(f'../../Dataset/{dataset}/valid/*/*.mid')
        test_files = glob.glob(f'../../Dataset/{dataset}/test/*/*.mid')

    elif args.dataset == 'pop1k7':
        files = glob.glob('../../Dataset/dataset/midi_transcribed/*/*.midi')

    elif args.dataset == 'ASAP':
        files = pickle.load(open('../../Dataset/ASAP_song.pkl', 'rb'))
        files = [f'../../Dataset/asap-dataset/{file}' for file in files]

    elif args.input_dir:
        files = glob.glob(f'{args.input_dir}/*.mid')

    else:
        print('not supported')
        exit(1)

    if args.dataset in {'pop909', 'emopia', 'pianist8'}:
        extract(train_files, args, model, 'train')
        extract(valid_files, args, model, 'valid')
        extract(test_files, args, model, 'test')
    else:
        # in one single file
        extract(files, args, model)


if __name__ == '__main__':
    main()
