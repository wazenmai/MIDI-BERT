import numpy as np
import pickle
import utils

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
        # pad word
        self.pad_word = [self.event2word[etype]['%s <PAD>' % etype] for etype in self.event2word]

    def extract_events(self, input_path, task):
        note_items, tempo_items = utils.read_items(input_path)
        note_items = utils.quantize_items(note_items)
        max_time = note_items[-1].end
        items = tempo_items + note_items
        
        groups = utils.group_items(items, max_time)
        events = utils.item2event(groups, task)
        return events

    def padding(self, data, max_len, ans):
        pad_len = max_len - len(data)
        #pad_tp = ['Bar <PAD>', 'Position <PAD>', 'Pitch <PAD>', 'Duration <PAD>']
        pad_tp = self.pad_word
        if not ans:
            for _ in range(pad_len):
                data.append(pad_tp)
        else:
            for _ in range(pad_len):
                data.append(0)

        return data

    def prepare_data(self, midi_paths, task, max_len):

        if task == 'melody' or task == 'velocity' or task == None:
            # extract events
            all_events = []
            for path in midi_paths:
                events = self.extract_events(path, task)
                all_events.append(events)

            # event to word
            all_words, all_ys = [], []
            for events in all_events:
                words, ys = [], []
                for note_tuple in events:
                    nts, to_class = [], -1
                    for e in note_tuple:
                        e_text = '{} {}'.format(e.name, e.value)
                        nts.append(self.event2word[e.name][e_text])
                        if e.name == 'Pitch':
                            to_class = e.Type
                    words.append(nts)
                    ys.append(to_class+1)

                # slice to chunks so that max_len = 512
                slice_words, slice_ys = [], []
                #max_len = 512
                for i in range(0, len(words), max_len):
                    slice_words.append(words[i:i+max_len])
                    slice_ys.append(ys[i:i+max_len])
                
                # padding
                if len(slice_words[-1]) < max_len:
                    slice_words[-1] = self.padding(slice_words[-1], max_len, ans=False)
                if len(slice_ys[-1]) < max_len:
                    slice_ys[-1] = self.padding(slice_ys[-1], max_len, ans=True)
                
                all_words = all_words + slice_words
                all_ys = all_ys + slice_ys
        
            all_words = np.array(all_words)
            all_ys = np.array(all_ys)
        else:
            all_words, all_ys = [], []
            for path in midi_paths:
                name = path.split('/')[-1].split('_')[0]
                # fix file name typo
                if task == "composer":
                    if name == "Hisaisi":
                        name = "Hisaishi"
                    if name == "Ryuici":
                        name = "Ryuichi"
                events = self.extract_events(path, task)
                words = []
                for note_tuple in events:
                    nts = []
                    for e in note_tuple:
                        # TODO: cut pitch limit
                        e_text = '{} {}'.format(e.name, e.value)
                        nts.append(self.event2word[e.name][e_text])
                    words.append(nts)
                
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
                elif task == "emotion" and len(slice_words[-1] < 512):
                    slice_words[-1] = self.padding(slice_words[-1], max_len, ans=False)
                
                all_words = all_words + slice_words
                all_ys = all_ys + slice_ys

        return all_words, all_ys
