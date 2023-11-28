import os
import pickle
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.colors import colorConverter
import miditoolkit.midi.parser as midparser
import setting

from analyzer import extract_melody
from melody_extraction.midibert.utils import DEFAULT_VELOCITY_BINS, DEFAULT_FRACTION, DEFAULT_DURATION_BINS, DEFAULT_TEMPO_INTERVALS, DEFAULT_RESOLUTION

e2w, w2e = pickle.load(open("data_creation/prepare_data/dict/CP.pkl", "rb"))
keys = list(w2e.keys())
ans_dict = {
    0: "padding",
    1: "melody", # melody & bridge
    2: "non-melody" # piano
}
bar_tick = 16
seg_len = 512

# DEFAULT_RESOLUTION = 480
# DEFAULT_VELOCITY_BINS = np.linspace(0,  128, 64+1, dtype=np.int)
ticks_per_beat = 480
ticks_per_bar = DEFAULT_RESOLUTION * 4 # assume 4/4

class NOTE(object):
    def __init__(self, start, end, pitch, velocity, Type):
        self.velocity = velocity
        self.pitch = pitch
        self.start = start
        self.end = end
        self.Type = Type
    def get_duration(self):
        return self.end - self.start
    def __repr__(self):
        return f'Note(start={self.start}, end={self.end}, pitch={self.pitch}, velocity={self.velocity}, Type={self.Type})'

def make_note_dict():
    note_dict = {}
    note = ['C', 'D', 'E', 'F', 'G', 'A', 'B']
    note_dict[21] = 'A0'
    note_dict[23] = 'B0'
    base_number = 24
    for i in range(1, 10):
        for n in note:
            note_dict[base_number] = n + str(i)
            if n == 'E' or n == 'B':
                base_number += 1
            else:
                base_number += 2
    return note_dict

def events2notes(events, verbose=True):
    notes = []
    if len(events.shape) > 2:
        events = events.reshape(-1, events.shape[-1])
    
    current_bar = -1
    for event in events:
        # 2.1 Turn CP token to events
        # [Bar, Position, Pitch, Velocity, Duration, Tempo]
        bar = w2e['Bar'][event[0]]
        pos = w2e['Position'][event[1]]
        pitch = w2e['Pitch'][event[2]]
        velocity = w2e['Velocity'][event[3]]
        duration = w2e['Duration'][event[4]]

        if bar == "Bar <PAD>":
            continue
        if verbose:
            print(bar, pos, pitch, velocity, duration)
        # 2.2 Turn events to value
        pos = int(pos.split(" ")[1].split("/")[0]) - 1
        pitch = int(pitch.split(" ")[1])
        duration = DEFAULT_DURATION_BINS[int(duration.split(" ")[1])]

        # 2.3 Turn value to note
        if bar == "Bar New":
            current_bar += 1
        current_bar_st = current_bar * ticks_per_bar
        current_bar_et = (current_bar + 1) * ticks_per_bar
        flags = np.linspace(current_bar_st, current_bar_et, DEFAULT_FRACTION, endpoint=False, dtype=int)
        st = flags[pos]
        et = st + duration
        notes.append(NOTE(st, et, pitch, velocity, 0))
        if verbose:
            print(st, et, pitch, velocity)
    return notes


def data2pr(data, ans, max_bar, verbose=False):
    
    # 24 bars = 24 * 4 (beat)* 4(semiquater) = 384 
    melody_pr = np.zeros((128, max_bar * 16))
    accomp_pr = np.zeros((128, max_bar * 16))
    
    bars = -1
    for i in range(data.shape[0]): # segment
        for j in range(data.shape[1]): # note
            # [Bar, Position, Pitch, Velocity, Duration, Tempo]
            if data[i][j][0] == 0:
                bars += 1
            if bars >= max_bar:
                break
            
            duration = data[i][j][4] # +1
            pitch = data[i][j][2] + 22
            pos = data[i][j][1]

            start = bars * bar_tick + pos
            duration = DEFAULT_DURATION_BINS[duration]
            duration = duration // 120
            end = min(start + duration, max_bar*16 - 1)

            if ans[i][j] == 1:
                melody_pr[pitch, start: end] =  np.ones((1, end - start))
            elif ans[i][j] == 2:
                accomp_pr[pitch, start: end] = np.ones((1, end - start))
            if verbose:
                print("{} {} {} {}".format(pos, pitch, duration, ans_dict[ans[i][j]]))
        if bars >= max_bar:
            break
    return melody_pr, accomp_pr

def plot_roll(pianoroll, pixels, filename):
    # pianoroll[idx, msg['pitch'], note_on_start_time:note_on_end_time] = intensity   # idx => 0: melody, 1: non-melody
    
    # build and set figure object
    plt.ioff()
    fig = plt.figure(figsize=(17, 11))
    a1 = fig.add_subplot(111)
    a1.axis("equal")
    a1.set_facecolor("white")
    a1.set_axisbelow(True)
    a1.yaxis.grid(color='gray', linestyle='dashed')
    
    # set colors
    channel_nb = 2
    transparent = colorConverter.to_rgba('white')
    colors = [mpl.colors.to_rgba('lightcoral'), mpl.colors.to_rgba('cornflowerblue')] 
    cmaps = [mpl.colors.LinearSegmentedColormap.from_list('my_cmap', [transparent, colors[i]], 128) for i in range(channel_nb)]

    # build color maps
    for i in range(channel_nb):
        cmaps[i]._init()
        # create your alpha array and fill the colormap with them
        alphas = np.linspace(0, 1, cmaps[i].N + 3)
        # create the _lut array, with rgba value
        cmaps[i]._lut[:, -1] = alphas

    label_name = ['melody', 'non-melody']
    a1.imshow(pianoroll[1], origin="lower", interpolation="nearest", cmap=cmaps[1], aspect='auto', label=label_name[1])
    a1.imshow(pianoroll[0], origin="lower", interpolation="nearest", cmap=cmaps[0], aspect='auto', label=label_name[0])
    note_dict = make_note_dict()

    # set scale and limit of axis
    interval = 64
    plt.xticks([i*interval for i in range(13)], [i*4 for i in range(13)])
    plt.yticks([(24 + (y)*12) for y in range(8)], [note_dict[24 + (y)*12] for y in range(8)])
    plt.ylim([36, 96]) # C2 to C8

    # show legend, and create a patch (proxy artist) for every color 
    patches = [ mpatches.Patch(color=colors[i], label=label_name[i] ) for i in range(channel_nb) ]
    # put those patched as legend-handles into the legend
    first_legend = plt.legend(handles=[patches[0]], loc=2, fontsize=40)
    ax = plt.gca().add_artist(first_legend)
    plt.legend(handles=[patches[1]], loc=1, fontsize=40)
    
    # save pianoroll to figure
    plt.xticks(fontsize=40)
    plt.yticks(fontsize=40)
    plt.xlabel("bars", fontsize=40)
    plt.ylabel("note name", fontsize=40)

    plt.savefig('pianoroll_' + filename)
    return
    
def skyline_from_midi(file):
    obj = midparser.MidiFile(file)
    melody = obj.instruments[0].notes
    bridge = obj.instruments[1].notes
    piano = obj.instruments[2].notes
    melody = [NOTE(i.start, i.end, i.pitch, i.velocity, 0) for i in melody]
    bridge = [NOTE(i.start, i.end, i.pitch, i.velocity, 1) for i in bridge]
    piano = [NOTE(i.start, i.end, i.pitch, i.velocity, 2) for i in piano]
    all_notes = []
    all_notes.extend(melody)
    all_notes.extend(bridge)
    all_notes.extend(piano)
    all_notes.sort(key=lambda x:(x.start))
    pred_m = extract_melody(all_notes)
    print(pred_m, len(pred_m))
    melody_pr = notes_to_pianoroll(pred_m, 48*4*4)
    accomp_pr = notes_to_pianoroll(all_notes, 48*4*4)
    return melody_pr, accomp_pr

def skyline_from_cp(data):
    """
    2023/11/26
    """
    notes = events2notes(data, verbose=False)
    pred_m = extract_melody(notes)
    
    melody_pr = notes_to_pianoroll(pred_m, 48*4*4) # 48 bars, 4 ticks per beat, 4 semiquaver per beat
    accomp_pr = notes_to_pianoroll(notes, 48*4*4)
    return melody_pr, accomp_pr
        

def notes_to_pianoroll(notes, length):
    melody_pr = np.zeros((128, length))
    
    for i, note in enumerate(notes):
        s, e = int(note.start/120), int(note.end/120)       # 4 pixel per beat
        print(s, e)
        if s > length:
            break
        e = min(length, e)
        melody_pr[note.pitch, s:e] = np.ones((1, e-s))

    return melody_pr

def main():
    data = np.load(setting.data_path)
    if 'gt' in setting.mode:
        # ground truth
        ans = np.load(setting.ground_truth_path)
        melody_pr, accomp_pr = data2pr(data, ans, 48, verbose=True)
        plot_roll([melody_pr, accomp_pr], 48*4*4, 'gt')
        print("-------")
    
    if 'skyline' in setting.mode:
        # skyline_from_midi
        # sky_melody, sky_accomp = skyline2("018.mid")
        # print(sky_melody.shape, sky_accomp.shape)
        # plot_roll([sky_melody, sky_accomp], 48*4*4)

        # skyline_from_cp 2023/11/26
        sky_melody, sky_accomp = skyline_from_cp(data)
        plot_roll([sky_melody, sky_accomp], 48*4*4, 'skyline')
        print("-------")
    
    if 'bert' in setting.mode:
        # bert
        bert_pred = np.load(setting.bert_ans_path)
        melody_pr, accomp_pr = data2pr(data, bert_pred, 48, verbose=True)
        plot_roll([melody_pr, accomp_pr], 48*4*4, 'bert')


if __name__ == '__main__':
    main()
