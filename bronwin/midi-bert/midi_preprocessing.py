import numpy as np

import miditoolkit
import single_utils
import pickle
import os

event2word, word2event = pickle.load(open('/home/wazenmai/NAS_189/home/MIDI-BERT/bert_exp/midi-bert/bert_dict.pkl', 'rb'))

# def load_vocab(data_path='home/wazenmai/practice/remi/bert_dict.pkl'):
#     return pickle.load(open(data_path, 'rb'))


class MidiTokenizer(object):
    def __init__(self, vocab_file=None):
        self.event2word, self.word2event = load_vocab(vocab_file)
    
    
    # def tokenize

def prepare_events(data_path, no_tempo):
    midi_obj = miditoolkit.midi.parser.MidiFile(data_path)
    note_items, tempo_items = single_utils.read_items(data_path)
    note_items = single_utils.quantize_items(note_items)
    items = tempo_items + note_items
    max_time = note_items[-1].end
    groups = single_utils.group_items(items, max_time)
    events = single_utils.item2event(groups)
    all_events = []
    all_events.extend(events) # have track information
    if no_tempo:
        delete_item = []
        for i, e in enumerate(all_events):
            if "Velocity" in e.name or "Tempo" in e.name:
                delete_item.append(i)
        delete_item.reverse()
        for d in delete_item:
            del all_events[d]
    return all_events

def prepare_data(data_path):
    midi_obj = miditoolkit.midi.parser.MidiFile(data_path)
    note_items, tempo_items = single_utils.read_items(data_path)
    note_items = single_utils.quantize_items(note_items)
    items = tempo_items + note_items
    max_time = note_items[-1].end
    groups = single_utils.group_items(items, max_time)
    events = single_utils.item2event(groups)
    all_events = []
    all_events.append(events) # have track information
    # event to word
    all_words = []
    for events in all_events:
        words = []
        for event in events:
            e = '{}_{}'.format(event.name, event.value)
            if e in event2word:
                words.append(event2word[e])
            else:
                # OOV
                if event.name == 'Note Velocity':
                    print('Note Velocity ', event.value)
                    # replace with max velocity based on our training data
                    words.append(event2word['Note Velocity_39'])
                else:
                    # something is wrong
                    # you should handle it for your own purpose
                    print('something is wrong! {}'.format(e))
        all_words.append(words)
    return all_words

def main():
    """
    dataset_path = '/home/wazenmai/NAS_189/home/datasets/POP909-Dataset/POP909'
    dataset = []

    i = 0
    for subdir, dirs, files in os.walk(dataset_path):
        for f in files:
            if i >= 100:
                break
            if '.mid' in f and not 'v' in f:
                filepath = subdir + os.sep + f
                if filepath not in small:
                    dataset.append(filepath)
                    i += 1
            
    print(i)
    # for path in dataset:
    #     print(path)
    with open('pop909_pretrain_data_eval.pkl', 'wb') as f:
        pickle.dump(dataset, f)
    """

    dataset_path = '/home/wazenmai/NAS_189/home/datasets/midi_bert/midi_bert_train_data_1562.pkl'
    dataset = pickle.load(open(dataset_path, 'rb'))
    b = []
    ev = prepare_events(dataset[5], no_tempo=True, once=True)
    # for data in dataset:
    #     events = prepare_events(data, no_tempo=True)
    #     b.append(events)
    

if __name__ == "__main__":
    main()
        
    

