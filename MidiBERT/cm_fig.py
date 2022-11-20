import matplotlib.pyplot as plt
import itertools
import numpy as np


def save_cm_fig(cm, classes, normalize, title, outdir, seq):
    if not seq:
        cm = cm[1:,1:]  # exclude padding

    if normalize:
        cm = cm.astype('float')*100/cm.sum(axis=1)[:,None]
   
#    print(cm)
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title(title)
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, fontsize=18)
    plt.yticks(tick_marks, classes, fontsize=18)

    fmt = '.2f' if normalize else 'd'
    threshold = cm.max()/2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i,j],fmt), fontsize=11,
                horizontalalignment='center',
                color='white' if cm[i,j] > threshold else 'black')
    plt.xlabel('predicted', fontsize=16)
    plt.ylabel('true', fontsize=16)
    plt.tight_layout()
    
    plt.savefig(f'{outdir}/cm_{title.split()[2]}.jpg')
    return
