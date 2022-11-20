import numpy as np
import pickle
from tqdm import tqdm
import data_creation.prepare_data.utils as utils

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

class CP(object):
    def __init__(self, dict):
        # load dictionary
        self.event2word, self.word2event = pickle.load(open(dict, 'rb'))
        # pad word: ['Bar <PAD>', 'Position <PAD>', 'Pitch <PAD>', 'Velocity <PAD>', 'Duration <PAD>', 'Tempo <PAD>']
        self.pad_word = [self.event2word[etype]['%s <PAD>' % etype] for etype in self.event2word]

    def extract_events(self, input_path, task):
        note_items, tempo_items = utils.read_items(input_path)
        if len(note_items) == 0:   # if the midi contains nothing
            return None
        note_items = utils.quantize_items(note_items)
        max_time = note_items[-1].end
        items = tempo_items + note_items
        
        groups = utils.group_items(items, max_time)
        events = utils.item2event(groups, task)
        return events

    def padding(self, data, max_len, ans):
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
            if not events:  # if midi contains nothing
                print(f'skip {path} because it is empty')
                continue
            # events to words
            words, ys = [], []
            for note_tuple in events:
                nts, to_class = [], -1
                for e in note_tuple:
                    # avoid the pitch out of the defined dictionary
                    if (e.name == 'Pitch' and e.value < 22):
                        e.value = 22
                    if (e.name == 'Pitch' and e.value >= 108):
                        e.value = 107
                    e_text = '{} {}'.format(e.name, e.value)
                    nts.append(self.event2word[e.name][e_text])
                    if e.name == 'Pitch':
                        to_class = e.Type
                words.append(nts)
                if task == 'melody' or task == 'velocity':
                    ys.append(to_class+1)

            # slice to chunks so that max length = max_len (default: 512)
            slice_words, slice_ys = [], []
            for i in range(0, len(words), max_len):
                slice_words.append(words[i:i+max_len])
                if task == "composer":
                    name = path.split('/')[-2]
                    slice_ys.append(Composer[name])
                elif task == "emotion":
                    name = path.split('/')[-1].split('_')[0]
                    slice_ys.append(Emotion[name])
                else:
                    slice_ys.append(ys[i:i+max_len])
            
            # padding or drop
            # drop only when the task is 'composer' and the data length < max_len//2
            if len(slice_words[-1]) < max_len:
                if task == 'composer' and len(slice_words[-1]) < max_len//2:
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

class REMI(object):
    def __init__(self, dict):
        # load dictionary
        self.event2word, self.word2event = pickle.load(open(dict, 'rb'))
        self.pad_word = self.event2word['Pad None'] 

    def extract_events(self, input_path, task):
        note_items, tempo_items = utils.read_items(input_path)
        assert len(note_items)
        note_items = utils.quantize_items(note_items)
        max_time = note_items[-1].end
        items = tempo_items + note_items
        groups = utils.group_items(items, max_time)
        events = utils.item2event_remi(groups, task)
        return events

    def padding(self, data, max_len, ans=False):
        pad_len = max_len - len(data)
        if not ans:
            for _ in range(pad_len):
                data.append(self.pad_word)
        else:
            for _ in range(pad_len):
                data.append(0)
        return data

    def prepare_data(self, midi_paths, task, max_len):
        print("task: ", task)
        if task == "melody" or task == "velocity":
            all_words, all_ys = [], []
            for path in midi_paths:
                print("path: ", path)
                events = self.extract_events(path, task)
                words, ys = [], []
                for event in events:
                    e = '{} {}'.format(event.name, event.value)
                    if e in self.event2word:
                        words.append(self.event2word[e])
                        ys.append(event.Type + 1) 
                    else:
                        # OOV
                        if event.name == 'Velocity':
                            # replace with max velocity based on our training data
                            words.append(self.event2word['Velocity 0'])
                            ys.append(0)
                        else:
                            # something is wrong
                            # you should handle it for your own purpose
                            print('something is wrong! {}'.format(e))
        
                # slice to chunks so that max_len = 512
                slice_words, slice_ys = [], []
                for i in range(0, len(words), max_len):
                    slice_words.append(words[i:i+max_len])
                    slice_ys.append(ys[i:i+max_len])
                
                # padding
                if len(slice_words[-1]) < max_len:
                    slice_words[-1] = self.padding(slice_words[-1], max_len, ans=False)
                    slice_ys[-1] = self.padding(slice_ys[-1], max_len, ans=True)
                
                all_words = all_words + slice_words
                all_ys = all_ys + slice_ys
        else:
            all_words, all_ys = [], []
            for path in midi_paths:
                print("path: ", path)
                name = path.split('/')[-1].split('_')[0]
                # fix file name typo
                if task == "composer":
                    if name == "Hisaisi":
                        name = "Hisaishi"
                    if name == "Ryuici":
                        name = "Ryuichi"
                events = self.extract_events(path, task)
                words = []
                for event in events:
                    e = '{} {}'.format(event.name, event.value)
                    if e in self.event2word:
                        words.append(self.event2word[e])
                    else:
                        print('something is wrong! {}'.format(e))
                slice_words, slice_ys = [], []
                for i in range(0, len(words), max_len):
                    slice_words.append(words[i:i + max_len])
                    if task == "composer":
                        slice_ys.append(Composer[name])
                    elif task == "emotion":
                         slice_ys.append(Emotion[name])
                
                # padding
                if task == 'composer' and len(slice_words[-1]) < 512:
                    if len(slice_words[-1]) < 256:
                        slice_words.pop()
                        slice_ys.pop()
                    else:
                        slice_words[-1] = self.padding(slice_words[-1], max_len, ans=False)
                elif len(slice_words[-1]) < 512:
                    slice_words[-1] = self.padding(slice_words[-1], max_len, ans=False)
                all_words.extend(slice_words)
                all_ys.extend(slice_ys)
        all_words = np.array(all_words)
        all_ys = np.array(all_ys)
        print(all_words.shape, all_words)
        return all_words, all_ys
