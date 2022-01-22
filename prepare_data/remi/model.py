import numpy as np
import pickle
import utils
from tqdm import tqdm

Composer = {
    "Bethel": 0,
    "Clayderman": 1,
    "Einaudi": 2,
    "Hancock": 3,
    "Hillsong": 4,
    "Hisaishi": 5,
    "Ryuichi": 6,
    "Yiruma": 7,
    "Padding": 8,
}

Emotion = {
    "Q1": 0,
    "Q2": 1,
    "Q3": 2,
    "Q4": 3,
}

class REMI(object):
    def __init__(self, dict):
        # load dictionary
        self.event2word, self.word2event = pickle.load(open(dict, 'rb'))
        self.pad_word = self.event2word['Pad_None'] 

    def extract_events(self, input_path, mode):
        note_items, tempo_items = utils.read_items(input_path)
        note_items = utils.quantize_items(note_items)
        max_time = note_items[-1].end
        items = tempo_items + note_items
        groups = utils.group_items(items, max_time)
        events = utils.item2event(groups, mode)
        return events

    def padding(self, data, max_len, ans=False):
        pad_len = max_len - len(data)
        for _ in range(pad_len):
            if not ans:
                data.append(self.pad_word)
            else:
                data.append(0)
        return data

    def prepare_data(self, midi_paths, task, max_len):
        all_words, all_ys = [], []

        for path in tqdm(midi_paths):
            # extract events
            events = self.extract_events(path, task)

            # events to words
            words, ys = [], []
            for event in events:
                e = '{}_{}'.format(event.name, event.value)
                if e in self.event2word:
                    words.append(self.event2word[e])
                    ys.append(event.Type + 1) 
                else:
                    # OOV
                    if event.name == 'Note Velocity':
                        # replace with max velocity based on our training data
                        words.append(self.event2word['Note Velocity_21'])
                        ys.append(0)
                    else:
                        # something is wrong
                        print('something is wrong! {}'.format(e))
        
            # slice to chunks so that max_len = 512
            slice_words, slice_ys = [], []

            for i in range(0, len(words), max_len):
                slice_words.append(words[i:i+max_len])
                if task == 'composer':
                    name = path.split('/')[-2]
                    slice_ys.append(Composer[name])
                elif task == 'emotion':
                    name = path.split('/')[-1].split('_')[0]
                    slice_ys.append(Emotion[name])
                else:
                    slice_ys.append(ys[i:i+max_len])
            
            # padding or drop
            if len(slice_words[-1]) < max_len:
                if task == 'composer' and len(slice_words[-1]) << max_len//2:
                    slice_words.pop()
                    slice_ys.pop()
                else:
                    slice_words[-1] = self.padding(slice_words[-1], max_len, ans=False)
            if (task == 'melody' or task == 'velocity') and len(slice_ys[-1]) < max_len:
                slice_ys[-1] = self.padding(slice_ys[-1], max_len, ans=True)
            
            all_words = all_words + slice_words
            all_ys = all_ys + slice_ys
            
        all_words = np.array(all_words)
        all_ys = np.array(all_ys)
            
        return all_words, all_ys
