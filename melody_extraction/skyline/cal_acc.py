from analyzer import extract_melody
import glob
from operator import itemgetter
import miditoolkit.midi.parser as midparser

class NOTE(object):
    def __init__(self, start, end, velocity, pitch, Type):
        self.start = start
        self.end = end
        self.velocity = velocity
        self.pitch = pitch
        self.Type = Type
    def __repr__(self):
        return 'NOTE(start={}, end={}, velocity={}, pitch={}, Type={})'.format(
            self.start, self.end, self.velocity, self.pitch, self.Type)


def extract(file, f):
    obj = midparser.MidiFile(file)
    melody = obj.instruments[0].notes
    bridge = obj.instruments[1].notes
    piano = obj.instruments[2].notes
    melody = [NOTE(i.start, i.end, i.velocity, i.pitch, 0) for i in melody]
    bridge = [NOTE(i.start, i.end, i.velocity, i.pitch, 1) for i in bridge]
    piano = [NOTE(i.start, i.end, i.velocity, i.pitch, 2) for i in piano]
    all_notes = []
    all_notes.extend(melody)
    all_notes.extend(bridge)
    all_notes.extend(piano)

    all_notes.sort(key=lambda x:(x.start))
    
    print('# melody: {}, # bridge: {}, # piano: {}'.format(len(melody), len(bridge), len(piano)))
    f.write('# melody: {}, # bridge: {}, # piano: {}\n'.format(len(melody), len(bridge), len(piano)))
    
    # extract
    pred_m = extract_melody(all_notes)
    
    all_notes = set(all_notes)
    pred_m = set(pred_m)

    TP1, TP2, FP1, FP2 = 0, 0, 0, 0
    for i in pred_m:
        if i.Type == 0 or i.Type == 1:
            # predict correctly
            TP1 += 1
        else:
            FP1 += 1
    
    FN1 = len(melody) + len(bridge) - TP1 
    TN1 = len(all_notes) - TP1 - FN1 - FP1

    for i in pred_m:
        if i.Type == 0:
            # predict correctly
            TP2 += 1
        else:
            FP2 += 1
    
    FN2 = len(melody) - TP2
    TN2 = len(all_notes) - TP2 - FP2 - FN2

    print('accuracy (melody only):', (TP2+TN2)/(TP2+TN2+FP2+FN2))
    print('accuracy (melody & bridge):', (TP1+TN1)/(TP1+TN1+FP1+FN1))

    f.write('TN2:{}, TP2:{}, FN2:{}, FP2:{}\n'.format(TN2, TP2, FN2, FP2))
    f.write('accuracy (melody only):' + str((TP2+TN2)/(TP2+TN2+FP2+FN2)) + '\n')

    f.write('TN1:{}, TP1:{}, FN1:{}, FP1:{}\n'.format(TN1, TP1, FN1, FP1))
    f.write('accuracy (melody & bridge):' + str((TP1+TN1)/(TP1+TN1+FP1+FN1)) + '\n')
    return TP1, TN1, FP1, FN1, TP2, TN2, FP2, FN2

def main():
    # use test set ONLY
    # please change the root_dir to `your path to pop909 test set`
    root_dir = '/home/user/Dataset/pop909_aligned/test'
    files = glob.glob(f'{root_dir}/*.mid')
    
    TP1, TN1, FP1, FN1, TP2, TN2, FP2, FN2 = 0,0,0,0,0,0,0,0

    with open('acc.log','a') as f:
        f.write('root_dir: ' + root_dir + '\n')
        f.write('# file: ' + str(len(files)) + '\n')
        for file in files:
            print('\n[',file,']')
            f.write('\n[ ' + file.split('/')[-1] + ' ]\n')
            tp1, tn1, fp1, fn1, tp2, tn2, fp2, fn2 = extract(file, f)
            TP1 += tp1; TN1 += tn1; FP1 += fp1; FN1 += fn1;
            TP2 += tp2; TN2 += tn2; FP2 += fp2; FN2 += fn2;
    
        f.write('\navg accuracy (melody only):' + str((TP2+TN2)/(TP2+TN2+FP2+FN2)) + '\n')
        f.write(f'(melody only) TP: {TP2}, FP: {FP2}, FN: {FN2}, TN:{TN2} \n')
        f.write(f'(melody & bridge) TP: {TP1}, FP: {FP1}, FN: {FN1}, TN:{TN1} \n')
        f.write(f'(melody & bridge): accuracy: {(TP1+TN1)/(TP1+TN1+FP1+FN1)}, precision: {TP1/(TP1+FP1)}, recall: {TP1/(TP1+FN1)}, f1_score: {2*TP1/(2*TP1+FP1+FN1)}\n')
        f.write('avg accuracy (melody & bridge):' + str((TP1+TN1)/(TP1+TN1+FP1+FN1)) + '\n')


if __name__ == '__main__':
    main()
