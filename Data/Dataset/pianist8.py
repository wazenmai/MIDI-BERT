import pickle
import os
import shutil

root = 'joann8512-Pianist8-ab9f541/'

def move(files, subset):
    for f in files:
        composer, piece = f.split('/')
        dest = os.path.join(root, subset, composer)
        os.makedirs(dest, exist_ok=True)

        src = os.path.join(root, 'midi', f)
        shutil.move(src, os.path.join(dest, piece))


if __name__ == '__main__':
    train = pickle.load(open('pianist8_train.pkl','rb'))
    valid = pickle.load(open('pianist8_valid.pkl','rb'))
    test = pickle.load(open('pianist8_test.pkl','rb'))

    move(train, 'train')
    move(valid, 'valid')
    move(test, 'test')

