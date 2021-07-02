from analyzer import extract_melody
import glob
import miditoolkit.midi.parser as midparser
from miditoolkit.midi import parser as mid_parser
from miditoolkit.pianoroll import parser as pr_parser
from pypianoroll import Track 
from matplotlib import pyplot as plt
import sys


def extract(file):
    obj = midparser.MidiFile(file)
    melody = obj.instruments[0].notes
    bridge = obj.instruments[1].notes
    all_notes = []
    all_notes.extend(melody)
    all_notes.extend(bridge)
    all_notes.extend(obj.instruments[2].notes)

    all_notes.sort(key=lambda x:(x.start))

    # extract
    pred = extract_melody(all_notes)
    MnB = melody+bridge
    all_notes, melody, pred, bridge, MnB = set(all_notes), set(melody), set(pred), set(bridge), set(MnB)
    print('# all notes: {}, # melody: {}, # bridge: {}, # pred_melody: {}'.format(len(all_notes), len(melody), len(bridge), len(pred)))

    intersection = melody.intersection(pred)
    intersection2 = MnB.intersection(pred)
    acc, acc2 = len(intersection), len(intersection2)
    sum1 = len(melody)+len(pred)-acc
    sum2 = len(melody)+len(bridge)+len(pred)-acc2
    print('accuracy (melody only):', acc/sum1)
    print('accuracy (melody & bridge):', acc2/sum2)
    return pred, MnB

def main():
    root_dir = '/home/yh1488/NAS-189/home/Dataset/pop909_aligned/'
    midi_ind = sys.argv[1]
    path = root_dir + midi_ind + '.mid'

    print('[',path,']')
    pred, truth = extract(path)
    
    roll = pr_parser.notes2pianoroll(pred)
    m_track = Track(name='predicted melody', program=0, is_drum=False, pianoroll=roll)
    t_roll = pr_parser.notes2pianoroll(truth)
    t_track = Track(name='predicted melody', program=0, is_drum=False, pianoroll=t_roll)
 
    filename = path.split('/')[-1].split('.')[0]
    #fig = m_track.plot()
    #plt.savefig(filename+'_pred.png')
    t_fig = t_track.plot()
    plt.savefig(filename+'_melody.png')


if __name__ == '__main__':
    main()
