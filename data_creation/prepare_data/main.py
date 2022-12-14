import os
from pathlib import Path
import glob
import pickle
import pathlib
import argparse
import numpy as np

from data_creation.prepare_data.model import *

def get_args():
    parser = argparse.ArgumentParser(description='')
    ### mode ###
    parser.add_argument('--task', default='', choices=['melody', 'velocity', 'composer', 'emotion'])
    parser.add_argument('--mode', choices=['cp', 'remi'], required=True)

    ### path ###
    parser.add_argument('--dict_dir', type=str, default='data_creation/prepare_data/dict')
    parser.add_argument('--dict', type=str, default='')
    parser.add_argument('--dataset', type=str, choices=["pop909", "pop1k7", "ASAP", "pianist8", "emopia"])
    parser.add_argument('--data_path', type=str, required=True, description="The path of dataset")
    parser.add_argument('--input_dir', type=str, default='')
    parser.add_argument('--input_file', type=str, default='')

    ### parameter ###
    parser.add_argument('--max_len', type=int, default=512)
    
    ### output ###    
    parser.add_argument('--output_dir', default="")
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
    elif args.dataset == None and args.input_dir == None and args.input_file == None:
        print('[error] Please specify the input directory or dataset')
        exit(1)

    if args.dict == '':
        args.dict = f'{args.dict_dir}/{args.mode}.pkl'

    if args.output_dir == '':
        args.output_dir = "Data/{args.mode}_data/tmp"

    return args


def extract(files, args, model, mode=''):
    '''
    files: list of midi path
    mode: 'train', 'valid', 'test', ''
    args.input_dir: '' or the directory to your custom data
    args.output_dir: the directory to store the data (and answer data) in CP representation
    '''
    assert len(files)

    print(f'Number of {mode} files: {len(files)}') 

    segments, ans = model.prepare_data(files, args.task, int(args.max_len))

    dataset = args.dataset if args.dataset != 'pianist8' else 'composer'

    if args.input_dir != '' or args.input_file != '':
        name = args.input_dir or args.input_file
        if args.name == '':
            args.name = Path(name).stem
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
    if (args.mode == 'cp'):
        model = CP(dict=args.dict)
    else:
        model = REMI(dict=args.dict)

    # if args.dataset == 'pop909':
    #     dataset = 'pop909_processed'
    # elif args.dataset == 'emopia':
    #     dataset = 'EMOPIA_1.0'
    # elif args.dataset == 'pianist8':
    #     dataset = 'joann8512-Pianist8-ab9f541'
    # elif args.dataset == 'ASAP':
    #     dataset = 'asap_dataset'
        

    if args.dataset == 'pop909':
        files = pickle.load(open("data_creation/preprocess_pop909/split.pkl", "rb"))
        train_files = [f'Data/Dataset/POP909-Dataset/POP909/{file.split(".")[0]}/{file}' for file in files["train_data"]]
        valid_files = [f'Data/Dataset/POP909-Dataset/POP909/{file.split(".")[0]}/{file}' for file in files["valid_data"]]
        test_files = [f'Data/Dataset/POP909-Dataset/POP909/{file.split(".")[0]}/{file}' for file in files["test_data"]]

    elif args.dataset == 'emopia':
        train_files = glob.glob(f'{args.data_path}/train/*.mid')
        valid_files = glob.glob(f'{args.data_path}/valid/*.mid')
        test_files = glob.glob(f'{args.data_path}/test/*.mid')

    elif args.dataset == "pianist8":
        train_files = glob.glob(f'{args.data_path}/train/*/*.mid')
        valid_files = glob.glob(f'{args.data_path}/valid/*/*.mid')
        test_files = glob.glob(f'{args.data_path}/test/*/*.mid')
        print(len(train_files))

    elif args.dataset == 'pop1k7':
        files = glob.glob('Data/Dataset/dataset/midi_transcribed/*/*.midi')

    elif args.dataset == 'ASAP':
        files = pickle.load(open('Data/Dataset/ASAP_song.pkl', 'rb'))
        files = [f'{args.data_path}/{file}' for file in files]
    
    elif args.input_dir:
        files = glob.glob(f'{args.input_dir}/*.mid')

    elif args.input_file:
        files = [args.input_file]

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
