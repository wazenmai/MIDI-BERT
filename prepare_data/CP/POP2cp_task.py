from modelCP_task import *
import numpy as np
import argparse
import os
import pathlib
import glob


def get_args():
    parser = argparse.ArgumentParser(description='')
    ### mode ###
    parser.add_argument('-t', '--task', choices=['melody', 'velocity'], required=True)

    ### output ###
    parser.add_argument('-d', '--dir', default="/home/yh1488/NAS-189/home/CP_data")
    parser.add_argument('-n', '--name', default="POP909cp")

    args = parser.parse_args()
    return args


def main(): 
    args = get_args()
    pathlib.Path(args.dir).mkdir(parents=True, exist_ok=True)
    
    # initialize model
    model = PopMusicTransformer(
        checkpoint='../my-checkpoint/CP.pkl',
        is_training=True)

    files = glob.glob('/home/yh1488/NAS-189/home/Dataset/pop909_aligned/*.mid') # not in order 
    print('number of files', len(files))

    segments, ans = model.prepare_data(files, args.task)

    print('segment shape', segments.shape)
    print('ans shape', ans.shape)

    #output_file = args.dir + '/' + args.name + '.npy'
    ans_file = args.dir+'/' + args.name + '_' + args.task[:3] + 'ans.npy'
    
    #if not os.path.exists(output_file):
    #np.save(output_file, segments)
    np.save(ans_file, ans)
    

if __name__ == '__main__':
    main()
