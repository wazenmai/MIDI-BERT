from modelCP_task import *
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
    parser.add_argument('--dict', default='../../BERT/dict/compact4/CP_cls.pkl')
    parser.add_argument('--dataset', choices=["pop909", "ailabs17k"], required=True)

    ### parameter ###
    parser.add_argument('--max_len', default=512)
    
    ### output ###    
    parser.add_argument('--dir', default="/home/yh1488/NAS-189/home/CP_data")
    parser.add_argument('-n', '--name', default="POP909cp")

    args = parser.parse_args()

    if args.task == 'melody' and args.dataset != 'pop909':
        print('[error] melody task is only supported for pop909 dataset')
        exit(1)
    return args


def main(): 
    args = get_args()
    pathlib.Path(args.dir).mkdir(parents=True, exist_ok=True)
    
    # initialize model
    model = PopMusicTransformer(
        checkpoint=args.dict,
        is_training=True)

    if args.dataset == 'pop909':
        files = glob.glob('/home/yh1488/NAS-189/home/Dataset/pop909_aligned/*.mid') # not in order 
    elif args.dataset == 'ailabs17k':
        files = glob.glob('/home/yh1488/NAS-189/home/Dataset/ailabs17k/*/*.mid')

    print('number of files', len(files))

    segments, ans = model.prepare_data(files, args.task, int(args.max_len))

    print('segment shape', segments.shape)
    output_file = args.dir + '/' + args.name + '.npy'
#    if not os.path.exists(output_file):
    np.save(output_file, segments)
    
    if args.task != None:
        print('ans shape', ans.shape)
        ans_file = args.dir+'/' + args.name + '_' + args.task[:3] + 'ans.npy'   
        np.save(ans_file, ans)
    

if __name__ == '__main__':
    main()
