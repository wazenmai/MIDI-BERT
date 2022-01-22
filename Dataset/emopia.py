import pickle
import os
import shutil

root = 'EMOPIA_1.0/'

def move(files, subset):
    for f in files:
        piece = f.split('/')[-1]
        src = os.path.join(root, 'midis', piece)
        shutil.move(src, os.path.join(root, subset, piece))


if __name__ == '__main__':
    train = pickle.load(open('emopia_train.pkl','rb'))
    valid = pickle.load(open('emopia_valid.pkl','rb'))
    test = pickle.load(open('emopia_test.pkl','rb'))

    dest = os.path.join(root, 'train')
    os.makedirs(dest, exist_ok=True)
    dest = os.path.join(root, 'valid')
    os.makedirs(dest, exist_ok=True)
    dest = os.path.join(root, 'test')
    os.makedirs(dest, exist_ok=True)

    move(train, 'train')
    move(valid, 'valid')
    move(test, 'test')
