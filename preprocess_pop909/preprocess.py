import miditoolkit
import os, pickle
from copy import deepcopy
import numpy as np
import glob
from exploratory import read_info_file
from tqdm import tqdm

from collections import Counter
from itertools import chain
import matplotlib.pyplot as plt

DEFAULT_TICKS_PER_BEAT = 480
DEFAULT_RESOLUTION = 120

root_dir = '../Dataset/POP909'
melody_out_dir = '../Dataset/pop909'

downbeat_records = []
all_bpms = []

def justify_tick(n_beats):
    n_ticks = n_beats * DEFAULT_TICKS_PER_BEAT
    return int(DEFAULT_RESOLUTION * round(n_ticks / DEFAULT_RESOLUTION))

def bpm2sec(bpm):
    return 60. / bpm

def calc_accum_secs(bpm, n_ticks, ticks_per_beat):
    return bpm2sec(bpm) * n_ticks / ticks_per_beat

def find_downbeat_idx_audio(audio_dbt):
    for st_idx in range(4):
        if audio_dbt[ st_idx ] == 1.:
            return st_idx

def get_note_time_sec(note, tempo_bpms, ticks_per_beat, tempo_change_ticks, tempo_accum_times):
    st_seg = np.searchsorted(tempo_change_ticks, note.start, side='left') - 1
    ed_seg = np.searchsorted(tempo_change_ticks, note.end, side='left') - 1
    # print (note.start, tempo_change_ticks[ st_seg ])

    start_sec = tempo_accum_times[ st_seg ] +\
                calc_accum_secs(
                    tempo_bpms[ st_seg ],
                    note.start - tempo_change_ticks[ st_seg ],
                    ticks_per_beat
                )
    end_sec = tempo_accum_times[ ed_seg ] +\
                calc_accum_secs(
                    tempo_bpms[ ed_seg ],
                    note.end - tempo_change_ticks[ ed_seg ],
                    ticks_per_beat
                )

    return start_sec, end_sec                        

def align_notes_to_secs(midi_obj):
    tempo_bpms = []
    tempo_change_ticks = []
    tempo_accum_times = []
    for tc in midi_obj.tempo_changes:
        # print (tc.tempo, tc.time)
        if tc.time == 0:
            tempo_accum_times.append( 0. )
        else:
            tempo_accum_times.append(
                tempo_accum_times[-1] + \
                calc_accum_secs(
                    tempo_bpms[-1], 
                    tc.time - tempo_change_ticks[-1], 
                    midi_obj.ticks_per_beat
                )
            )

        tempo_bpms.append(tc.tempo)
        tempo_change_ticks.append(tc.time)

    # print (tempo_accum_times)

    vocal_notes = []
    for note in midi_obj.instruments[0].notes:
        note_st_sec, note_ed_sec = get_note_time_sec(
                                        note, tempo_bpms,
                                        midi_obj.ticks_per_beat, tempo_change_ticks,
                                        tempo_accum_times
                                    )
        # print (note_st_sec, note_ed_sec)
        vocal_notes.append(
            {'st_sec': note_st_sec, 'ed_sec': note_ed_sec, 'pitch': note.pitch, 'velocity': note.velocity}
        )

    bridge_notes = []
    for note in midi_obj.instruments[1].notes:
        note_st_sec, note_ed_sec = get_note_time_sec(
                                        note, tempo_bpms,
                                        midi_obj.ticks_per_beat, tempo_change_ticks,
                                        tempo_accum_times
                                    )
        # print (note_st_sec, note_ed_sec)
        bridge_notes.append(
            {'st_sec': note_st_sec, 'ed_sec': note_ed_sec, 'pitch': note.pitch, 'velocity': note.velocity}
        )
    
    piano_notes = []
    for note in midi_obj.instruments[2].notes:
        note_st_sec, note_ed_sec = get_note_time_sec(
                                        note, tempo_bpms,
                                        midi_obj.ticks_per_beat, tempo_change_ticks,
                                        tempo_accum_times
                                    )
        # print (note_st_sec, note_ed_sec)
        piano_notes.append(
            {'st_sec': note_st_sec, 'ed_sec': note_ed_sec, 'pitch': note.pitch, 'velocity': note.velocity}
        )
    
    return vocal_notes, bridge_notes, piano_notes

def group_notes_per_beat(notes, beat_times):
    n_beats = len(beat_times)
    note_groups = [[] for _ in range(n_beats)]
    cur_beat = 0

    notes = sorted(notes, key=lambda x: (x['st_sec'], -x['pitch']))
        
    for note in notes:
        while cur_beat < (n_beats - 1) and note['st_sec'] > beat_times[ cur_beat + 1 ]:
            # print (cur_beat, note['st_sec'], beat_times[ cur_beat + 1 ]) 
            cur_beat += 1

        if cur_beat == 0 and note['st_sec'] < beat_times[0]:
            if note['st_sec'] >= (beat_times[0] - 0.1) and note['ed_sec'] - note['st_sec'] > 0.2:
                note['st_sec'] = beat_times[0]
            else:
                continue

        if cur_beat == n_beats - 1:
            if note['st_sec'] - beat_times[-1] > beat_times[-1] - beat_times[-2]:
                continue

        note_groups[ cur_beat ].append( deepcopy(note) )

    return note_groups

def remove_piano_notes_collision(vocal_notes, piano_notes):
    n_beats = len(vocal_notes)

    for beat in range(n_beats):
        if (beat - 1 >= 0 and len(vocal_notes[ beat - 1 ])) or \
             len (vocal_notes[ beat ]) or \
             (beat + 1 < n_beats and len(vocal_notes[ beat + 1 ])) or \
             (beat + 2 < n_beats and len(vocal_notes[ beat + 2 ])):
            piano_notes[ beat ] = []

    return piano_notes

def quantize_notes(notes, beat_times, downbeat_idx):
    quantized = [[] for _ in range(len(beat_times))]

    if downbeat_idx == 1:
        cur_tick = 3 * DEFAULT_TICKS_PER_BEAT
    elif downbeat_idx == 2:
        cur_tick = 2 * DEFAULT_TICKS_PER_BEAT
    elif downbeat_idx == 3:
        cur_tick = DEFAULT_TICKS_PER_BEAT
    else:
        cur_tick = 0

    for b_idx, beat_notes in enumerate(notes):
        beat_dur = beat_times[b_idx + 1] - beat_times[b_idx]\
                    if b_idx < len(notes) - 1 else beat_times[-1] - beat_times[-2]
        beat_st_sec = beat_times[b_idx]
    
        for note in beat_notes:
            note_dur_tick = justify_tick( (note['ed_sec'] - note['st_sec']) / beat_dur )
            if note_dur_tick == 0:
                continue
            note_st_tick = cur_tick +\
                         justify_tick( (note['st_sec'] - beat_st_sec) / beat_dur )

            if note_st_tick < 0:
                # print (note['st_sec'], beat_st_sec, b_idx, cur_tick)
                print ('[violation]', note_st_tick)

            note['st_tick'] = note_st_tick
            note['dur_tick'] = note_dur_tick
            quantized[ b_idx ].append( deepcopy(note) )

        cur_tick += DEFAULT_TICKS_PER_BEAT

    return quantized


def merge_notes(vocal_notes, bridge_notes, piano_notes):
    vocal_notes = list(chain(*vocal_notes))
    bridge_notes = list(chain(*bridge_notes))
    piano_notes = list(chain(*piano_notes))

    vocal_notes = sorted(
                        vocal_notes, 
                        key=lambda x : (x['st_tick'], -x['pitch'])
                    )
    bridge_notes = sorted(
                        bridge_notes, 
                        key=lambda x : (x['st_tick'], -x['pitch'])
                    )
    piano_notes = sorted(
                        piano_notes, 
                        key=lambda x : (x['st_tick'], -x['pitch'])
                    )

    return vocal_notes, bridge_notes, piano_notes


def dump_melody_midi(vocal_notes, bridge_notes, piano_notes, bpm_changes, midi_out_path):
    midi_obj = miditoolkit.midi.MidiFile()
    midi_obj.time_signature_changes = [
        miditoolkit.midi.containers.TimeSignature(4, 4, 0)
    ]
    midi_obj.tempo_changes = bpm_changes
    midi_obj.instruments = [
        miditoolkit.midi.Instrument(0, name='vocal'),
        miditoolkit.midi.Instrument(1, name='bridge'),
        miditoolkit.midi.Instrument(2, name='piano'),
    ]

    for n in vocal_notes:
        midi_obj.instruments[0].notes.append(
             miditoolkit.midi.containers.Note(
                 n['velocity'], n['pitch'], n['st_tick'], n['st_tick'] + n['dur_tick']
             )
        )

    for n in bridge_notes:
        midi_obj.instruments[1].notes.append(
             miditoolkit.midi.containers.Note(
                 n['velocity'], n['pitch'], n['st_tick'], n['st_tick'] + n['dur_tick']
             )
        )
    
    for n in piano_notes:
        midi_obj.instruments[2].notes.append(
             miditoolkit.midi.containers.Note(
                 n['velocity'], n['pitch'], n['st_tick'], n['st_tick'] + n['dur_tick']
             )
        )
    midi_obj.dump(midi_out_path)
    return

def align_midi_beats(piece_dir, subfolder):
    audio_beat_path = os.path.join(piece_dir, 'beat_audio.txt')
    midi_beat_times = read_info_file(audio_beat_path, [0])[0]
    midi_beat_idx = read_info_file(audio_beat_path, [1])[1]
    
    # find the 1st down beat
    downbeat_idx = find_downbeat_idx_audio(midi_beat_idx) 
    
    midi_obj = miditoolkit.midi.MidiFile(
                            os.path.join(root_dir, pdir, pdir + '.mid')
                        )

    vocal_notes, bridge_notes, piano_notes = align_notes_to_secs(midi_obj)
    
    vocal_notes = group_notes_per_beat(vocal_notes, midi_beat_times)
    bridge_notes = group_notes_per_beat(bridge_notes, midi_beat_times)
    piano_notes = group_notes_per_beat(piano_notes, midi_beat_times)
    
    vocal_notes = quantize_notes(vocal_notes, midi_beat_times, downbeat_idx)
    bridge_notes = quantize_notes(bridge_notes, midi_beat_times, downbeat_idx)
    piano_notes = quantize_notes(piano_notes, midi_beat_times, downbeat_idx)
    
    vocal_notes, bridge_notes, piano_notes = merge_notes(vocal_notes, bridge_notes, piano_notes)
    
    
    # recalculate bpm
    # change index 0 to 4
    if downbeat_idx == 0:
        downbeat_idx = 4
    
    first_beat_tick = (4-downbeat_idx) * DEFAULT_TICKS_PER_BEAT
    first_bpm = np.round( (60./(midi_beat_times[1]-midi_beat_times[0])), 2 )
    bpm_changes = [ miditoolkit.midi.containers.TempoChange(first_bpm, 0) ]

    beat_diff = [np.round(j-i, 2) for i,j in zip(midi_beat_times[:-1], midi_beat_times[1:])]
    bd_tmp = beat_diff[0]
    
    for i, bd in enumerate(beat_diff[1:]):
        if abs(bd-bd_tmp) > 0.05:
                bd_tmp = bd
                neighbor_avg = (beat_diff[i-1] + bd + beat_diff[i+1])/3
                _bpm = 60. / (bd)
                _st = first_beat_tick + ((i+2) * DEFAULT_TICKS_PER_BEAT)
                bpm_changes.append(miditoolkit.midi.containers.TempoChange(_bpm,_st))

        
    dump_melody_midi(
        vocal_notes,
        bridge_notes,
        piano_notes,
        bpm_changes,
        os.path.join(melody_out_dir, subfolder, piece_dir.split('/')[-1] + '.mid')
    )


if __name__ == '__main__':
    pieces_dir = pickle.load(open('qual_pieces.pkl', 'rb'))
    data_split = pickle.load(open('split.pkl','rb'))
    train_data = set(data_split['train_data'])
    valid_data = set(data_split['valid_data'])
    test_data = set(data_split['test_data'])

    os.makedirs(f'{melody_out_dir}/train', exist_ok=True)
    os.makedirs(f'{melody_out_dir}/valid', exist_ok=True)
    os.makedirs(f'{melody_out_dir}/test', exist_ok=True)
    
    for pdir in tqdm(pieces_dir):
        piece_dir = os.path.join(root_dir, pdir)
        mid = f'{pdir}.mid'
        if mid in train_data:
            subfolder = 'train'
        elif mid in valid_data:
            subfolder = 'valid'
        elif mid in test_data:
            subfolder = 'test'
        else:
            print(f'invalid midi {mid}')
            exit(1)
        align_midi_beats(piece_dir, subfolder)
    print(f"Preprocessed data saved at {melody_out_dir}")
