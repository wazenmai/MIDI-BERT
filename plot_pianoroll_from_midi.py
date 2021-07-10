from miditoolkit.midi import parser as mid_parser
from miditoolkit.pianoroll import parser as pr_parser
from pypianoroll import Track 
from matplotlib import pyplot as plt
import sys

def note2roll(path):
    obj = mid_parser.MidiFile(path)
    melody = obj.instruments[0].notes
    bridge = obj.instruments[1].notes
    piano = obj.instruments[2].notes

    m_roll = pr_parser.notes2pianoroll(melody)
    b_roll = pr_parser.notes2pianoroll(bridge)
    p_roll = pr_parser.notes2pianoroll(piano)

    return m_roll, b_roll, p_roll

def main():
    path = sys.argv[1]
    m, b, p = note2roll(path)
    m_track = Track(name='melody', program=0, is_drum=False, pianoroll=m)
    b_track = Track(name='bridge', program=0, is_drum=False, pianoroll=b)
    p_track = Track(name='piano', program=0, is_drum=False, pianoroll=p)
 
    filename = path.split('/')[-1]
    fig = m_track.plot()
    plt.savefig(filename+'_melody.png')
    fig = b_track.plot()
    plt.savefig(filename+'_bridge.png')
    fig = p_track.plot()
    plt.savefig(filename+'_piano.png')

    
if __name__ == '__main__':
    if len(sys.argv)!=2:
        print('Usage: python3 plot.py [midi_path]')
        exit(1)
    main()
