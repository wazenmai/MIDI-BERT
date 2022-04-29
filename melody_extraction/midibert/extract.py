"""
[ Melody Extraction ]
Given path to input midi file, save the predicted melody midi file. 
Please note that the model is trained on pop909 dataset (containing 3 classes: melody, bridge, accompaniment), 
so there are 2 interpretations: view `bridge` as `melody` or view it as `accompaniment`.
You could choose the mode - `bridge` is viewed as `melody` by default.

Also, the sequence is zero-padded so that the shape (length) is the same, but it won't affect the results, 
as zero-padded tokens will be excluded in post-processing.
"""

import argparse
import numpy as np
import random
import pickle
import os
import copy
import shutil
import json
import miditoolkit

from torch.utils.data import DataLoader
import torch
import torch.nn as nn
from transformers import BertConfig

from melody_extraction.midibert.midi2CP import CP
from melody_extraction.midibert.utils import DEFAULT_VELOCITY_BINS, DEFAULT_FRACTION, DEFAULT_DURATION_BINS, DEFAULT_TEMPO_INTERVALS, DEFAULT_RESOLUTION
from MidiBERT.model import MidiBert
from MidiBERT.finetune_model import TokenClassification


def boolean_string(s):
    if s not in ['False', 'True']:
        raise ValueError('Not a valid boolean string')
    return s == 'True'

def get_args():
    parser = argparse.ArgumentParser(description='')

    ### path setup ###
    parser.add_argument('--input_path', type=str, required=True, help="path to the input midi file")
    parser.add_argument('--output_path', type=str, default=None, help="path to the output midi file")
    parser.add_argument('--dict_file', type=str, default='data_creation/prepare_data/dict/CP.pkl')
    parser.add_argument('--ckpt', type=str, default='')

    ### parameter setting ###
    parser.add_argument('--max_seq_len', type=int, default=512, help='all sequences are padded to `max_seq_len`')
    parser.add_argument('--hs', type=int, default=768)
    parser.add_argument('--bridge', default=True, type=boolean_string, help='View bridge as melody (True) or accompaniment (False)')
    
    ### cuda ###
    parser.add_argument('--cpu', action="store_true")  # default: false

    args = parser.parse_args()

    root = 'result/finetune/'
    args.ckpt = root + 'melody_default/model_best.ckpt' if args.ckpt=='' else args.ckpt

    if not args.output_path:
        basename = args.input_path.split('/')[-1].split('.')[0]
        args.output_path = f'{basename}_melody.mid'

    return args


def load_model(args, e2w, w2e):
    print("\nBuilding BERT model")
    configuration = BertConfig(max_position_embeddings=args.max_seq_len,
                                position_embedding_type='relative_key_query',
                                hidden_size=args.hs)

    midibert = MidiBert(bertConfig=configuration, e2w=e2w, w2e=w2e)
    
    model = TokenClassification(midibert, 4, args.hs)
        
    print('\nLoading ckpt from', args.ckpt)  
    checkpoint = torch.load(args.ckpt, map_location='cpu')

    # remove module
    #from collections import OrderedDict
    #new_state_dict = OrderedDict()
    #for k, v in checkpoint['state_dict'].items():
    #    name = k[7:]
    #    new_state_dict[name] = v
    #model.load_state_dict(new_state_dict)
    model.load_state_dict(checkpoint['state_dict'])

    return model


def inference(model, tokens, pad_CP, device):
    """
        Given `model`, `tokens` (input), `pad_CP` (to indicate which notes are padded)
        Return inference output
    """
    tokens = torch.from_numpy(tokens).to(device)
    pad_CP = torch.tensor(pad_CP).to(device)
    attn = torch.all(tokens != pad_CP, dim=2).float().to(device)

    # forward (input, attn, layer idx)
    pred = model.forward(tokens, attn, -1)                      # pred: (batch, seq_len, class_num)
    output = np.argmax(pred.cpu().detach().numpy(), axis=-1)   # (batch, seq_len)

    return torch.from_numpy(output)


def get_melody_events(events, inputs, preds, pad_CP, bridge=True):
    """
        Filter out predicted melody events.
        Arguments:
        - events: complete events, including tempo changes and velocity
        - inputs: input compact_CP tokens (batch, seq, CP_class), np.array
        - preds: predicted classes (batch, seq), torch.tensor
            Note for predictions: 1 is melody, 2 is bridge, 3 is piano/accompaniment
        - pad_CP: padded CP representation (list)
        - bridge (bool): whether bridge is viewed as melody
    """
    numClass = inputs.shape[-1]
    inputs = inputs.reshape(-1, numClass)
    preds = preds.reshape(-1)
    pad_CP = np.array(pad_CP)

    melody_events = []
    note_ind = 0
    for event in events:
        if len(event) == 5:     # filter out melody events
            is_melody = preds[note_ind] == 1 or (bridge and preds[note_ind] == 2)
            is_valid_note = np.all(inputs[note_ind] != pad_CP)
            if is_valid_note and is_melody:
                melody_events.append(event)
            note_ind += 1
        else:
            melody_events.append(event)

    return melody_events


def events2midi(events, output_path, prompt_path=None):
    """
        Given melody events, convert back to midi
    """
    temp_notes, temp_tempos = [], []

    for event in events:
        if len(event) == 1:         # [Bar]
            temp_notes.append('Bar')
            temp_tempos.append('Bar')

        elif len(event) == 5:       # [Bar, Position, Pitch, Duration, Velocity]
            # start time and end time from position
            position = int(event[1].value.split('/')[0]) - 1
            # pitch
            pitch = int(event[2].value)
            # duration
            index = int(event[3].value)
            duration = DEFAULT_DURATION_BINS[index]
            # velocity
            index = int(event[4].value)
            velocity = int(DEFAULT_VELOCITY_BINS[index])
            # adding
            temp_notes.append([position, velocity, pitch, duration])

        else:                       # [Position, Tempo Class, Tempo Value]
            position = int(event[0].value.split('/')[0]) - 1
            if event[1].value == 'slow':
                tempo = DEFAULT_TEMPO_INTERVALS[0].start + int(event[2].value)
            elif event[1].value == 'mid':
                tempo = DEFAULT_TEMPO_INTERVALS[1].start + int(event[2].value)
            elif event[1].value == 'fast':
                tempo = DEFAULT_TEMPO_INTERVALS[2].start + int(event[2].value)
            temp_tempos.append([position, tempo])

    # get specific time for notes
    ticks_per_beat = DEFAULT_RESOLUTION
    ticks_per_bar = DEFAULT_RESOLUTION * 4 # assume 4/4
    notes = []
    current_bar = 0
    for note in temp_notes:
        if note == 'Bar':
            current_bar += 1
        else:
            position, velocity, pitch, duration = note
            # position (start time)
            current_bar_st = current_bar * ticks_per_bar
            current_bar_et = (current_bar + 1) * ticks_per_bar
            flags = np.linspace(current_bar_st, current_bar_et, DEFAULT_FRACTION, endpoint=False, dtype=int)
            st = flags[position]
            # duration (end time)
            et = st + duration
            notes.append(miditoolkit.Note(velocity, pitch, st, et))

    # get specific time for tempos
    tempos = []
    current_bar = 0
    for tempo in temp_tempos:
        if tempo == 'Bar':
            current_bar += 1
        else:
            position, value = tempo
            # position (start time)
            current_bar_st = current_bar * ticks_per_bar
            current_bar_et = (current_bar + 1) * ticks_per_bar
            flags = np.linspace(current_bar_st, current_bar_et, DEFAULT_FRACTION, endpoint=False, dtype=int)
            st = flags[position]
            tempos.append([int(st), value])
    # write
    if prompt_path:
        midi = miditoolkit.midi.parser.MidiFile(prompt_path)
        #
        last_time = DEFAULT_RESOLUTION * 4 * 4
        # note shift
        for note in notes:
            note.start += last_time
            note.end += last_time
        midi.instruments[0].notes.extend(notes)
        # tempo changes
        temp_tempos = []
        for tempo in midi.tempo_changes:
            if tempo.time < DEFAULT_RESOLUTION*4*4:
                temp_tempos.append(tempo)
            else:
                break
        for st, bpm in tempos:
            st += last_time
            temp_tempos.append(miditoolkit.midi.containers.TempoChange(bpm, st))
        midi.tempo_changes = temp_tempos
    else:
        midi = miditoolkit.midi.parser.MidiFile()
        midi.ticks_per_beat = DEFAULT_RESOLUTION
        # write instrument
        inst = miditoolkit.midi.containers.Instrument(0, is_drum=False)
        inst.notes = notes
        midi.instruments.append(inst)
        # write tempo
        tempo_changes = []
        for st, bpm in tempos:
            tempo_changes.append(miditoolkit.midi.containers.TempoChange(bpm, st))
        midi.tempo_changes = tempo_changes
    
    # write  
    midi.dump(output_path)
    print(f"predicted melody midi file is saved at {output_path}")

    return 


def main():
    args = get_args()
    with open(args.dict_file, 'rb') as f:
        e2w, w2e = pickle.load(f)

    compact_classes = ['Bar', 'Position', 'Pitch', 'Duration']
    pad_CP = [e2w[subclass][f"{subclass} <PAD>"] for subclass in compact_classes]

    # preprocess input file
    CP_model = CP(dict=args.dict_file)
    events, tokens = CP_model.prepare_data(args.input_path, args.max_seq_len)      # files, task, seq_len
    filename = args.input_path.split('/')[-1]
    print(f"'{filename}' is preprocessed to CP repr. with shape {tokens.shape}")
    
    # load pre-trained model
    model = load_model(args, e2w, w2e)

    # inference
    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else 'cpu')
    print("Using", device)
    predictions = inference(model, tokens, pad_CP, device)
    print(f"predicted melody shape {predictions.shape}")
    #np.save("input.npy", tokens)
    #np.save("pred.npy", predictions)
  
    # post-process    
    melody_events = get_melody_events(events, tokens, predictions, pad_CP, bridge=args.bridge)
    print(f"Melody Events: {len(melody_events)}/{len(events)}")

    # save melody midi
    melody_midi = events2midi(melody_events, args.output_path)


if __name__ == '__main__':
    main()
