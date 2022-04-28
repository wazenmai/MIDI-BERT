import numpy as np
import pickle
from tqdm import tqdm
import melody_extraction.midibert.utils as utils


class CP(object):
    def __init__(self, dict):
        # load dictionary
        self.event2word, self.word2event = pickle.load(open(dict, 'rb'))
        # pad word: ['Bar <PAD>', 'Position <PAD>', 'Pitch <PAD>', 'Duration <PAD>']
        classes = ['Bar', 'Position', 'Pitch', 'Duration']
        self.pad_word = [self.event2word[etype]['%s <PAD>' % etype] for etype in classes]

    def extract_events(self, input_path):
        note_items, tempo_items = utils.read_items(input_path)
        if len(note_items) == 0:   # if the midi contains nothing
            return None
        note_items = utils.quantize_items(note_items)
        max_time = note_items[-1].end
        items = tempo_items + note_items
        
        groups = utils.group_items(items, max_time)
        events = utils.item2event(groups)
        return events

    def padding(self, data, max_len):
        pad_len = max_len - len(data)
        for _ in range(pad_len):
            data.append(self.pad_word)

        return data

    def prepare_data(self, midi_path, max_len):
        """
            Prepare data for a single midi 
        """
        # extract events
        events = self.extract_events(midi_path)
        if not events:  # if midi contains nothing
            raise ValueError(f'The given {midi_path} is empty')

        # events to words
        # 1. Bar, Position, Pitch, Duration, Velocity ---> we only convert note events to words
        # 2. Position, Tempo Style, Tempo Class
        # 3. Bar
        words = []

        for tup in events:
            nts = []
            if len(tup) == 5:    # Note
                for e in tup:
                    if e.name == 'Velocity':
                        continue
                    e_text = '{} {}'.format(e.name, e.value)
                    nts.append(self.event2word[e.name][e_text])
                words.append(nts)
                
        # slice to chunks so that max length = max_len (default: 512)
        slice_words = []
        for i in range(0, len(words), max_len):
            slice_words.append(words[i:i+max_len])

        # padding or drop
        if len(slice_words[-1]) < max_len:
            slice_words[-1] = self.padding(slice_words[-1], max_len)

        slice_words = np.array(slice_words)

        return events, slice_words
