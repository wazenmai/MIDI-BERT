from model import *
import numpy as np
import sys
import glob


def main(): 
    if len(sys.argv)!=2:
        print("Usage: python3 remi1700.py <outputFile>")  
        exit(1)

    root = "/home/yh1488/NAS-189/homes/wazenmai/datasets/remi_1700/midi_transcribed"
    remi1700 = glob.glob(root + '/*/*.midi')

    print('remi1700 data:',len(remi1700))

    root = '/home/yh1488/NAS-189/home/remi_data/'
    outputFile = root+sys.argv[1]
    
    print("\nInitializing model...")
    # initialize model
    model = PopMusicTransformer(
        checkpoint='my-checkpoint/remi.pkl',
        is_training=True)
   
    segments = model.prepare_data(remi1700)
    seg = np.array(segments, dtype=object) 
    np.save(outputFile, seg)


if __name__ == '__main__':
    main()
