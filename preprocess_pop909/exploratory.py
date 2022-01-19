import miditoolkit
import os, pickle
import matplotlib.pyplot as plt
from collections import Counter

root_dir = '../Dataset/POP909'

def read_info_file(fpath, tgt_cols):
    with open(fpath, 'r') as f:
        lines = f.read().splitlines()

    ret_dict = {col: [] for col in tgt_cols}
    for l in lines:
        l = l.split()
        for col in tgt_cols:
            ret_dict[col].append( float(l[col]) )

    return ret_dict

if __name__ == '__main__':
    pieces_dir = [ i for i in os.listdir(root_dir) 
                    if os.path.isdir( os.path.join(root_dir, i) ) ]

    qualified_quad = 0
    qualified_triple = 0
    qualified_pieces = []

    for pdir in pieces_dir:
        audio_beat_path = os.path.join(root_dir, pdir, 'beat_audio.txt')
        audio_beats = read_info_file(audio_beat_path, [1])[1]

        midi_beat_path = os.path.join(root_dir, pdir, 'beat_midi.txt')
        midi_beats = read_info_file(midi_beat_path, [1, 2])
        midi_beats_minor, midi_beats_major = midi_beats[1], midi_beats[2]

        if max(audio_beats) == 4.:
            try:
                assert abs(0.25 - sum(midi_beats_major) / len(midi_beats_major)) < 0.03
                qualified_quad += 1
                qualified_pieces.append(pdir)
            except:
                print (pdir, '[error] 4-beat !!')
        elif max(audio_beats) == 3.:
            try:
                assert abs(0.33 - sum(midi_beats_minor) / len(midi_beats_major)) < 0.03, sum(midi_beats_minor) / len(midi_beats_major)
                qualified_triple += 1
            except:
                print (pdir, '[error] 3-beat !!')


    print('qualified quad: {}; qualified triple: {}'.format(qualified_quad, qualified_triple))
    
    pickle.dump(
        qualified_pieces, 
        open('qual_pieces.pkl', 'wb'), 
        protocol=pickle.HIGHEST_PROTOCOL
    )
