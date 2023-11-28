import numpy as np
import miditoolkit
import copy

# parameters for input
LABEL_VELOCITY_BINS = np.array([ 0, 32, 48, 64, 80, 96, 128])     # np.linspace(0, 128, 64+1, dtype=np.int)
DEFAULT_VELOCITY_BINS = np.linspace(0,  128, 64+1, dtype=np.int)
DEFAULT_FRACTION = 16
DEFAULT_DURATION_BINS = np.arange(60, 3841, 60, dtype=int)
DEFAULT_TEMPO_BINS = np.linspace(32, 224, 64+1, dtype=int)

# parameters for output
DEFAULT_RESOLUTION = 480

# define "Item" for general storage
class Item(object):
    def __init__(self, name, start, end, velocity, pitch, Type):
        self.name = name
        self.start = start
        self.end = end
        self.velocity = velocity
        self.pitch = pitch
        self.Type = Type

    def __repr__(self):
        return 'Item(name={}, start={}, end={}, velocity={}, pitch={}, Type={})'.format(
            self.name, self.start, self.end, self.velocity, self.pitch, self.Type)

# read notes and tempo changes from midi (assume there is only one track)
def read_items(file_path):
    midi_obj = miditoolkit.midi.parser.MidiFile(file_path)
    # note
    note_items = []
    num_of_instr = len(midi_obj.instruments) 
    
    for i in range(num_of_instr):
        notes = midi_obj.instruments[i].notes
        notes.sort(key=lambda x: (x.start, x.pitch))

        for note in notes:
            note_items.append(Item(
                name='Note',
                start=note.start, 
                end=note.end, 
                velocity=note.velocity, 
                pitch=note.pitch,
                Type=i))
                
    note_items.sort(key=lambda x: x.start)
    
    # tempo
    tempo_items = []
    for tempo in midi_obj.tempo_changes:
        tempo_items.append(Item(
            name='Tempo',
            start=tempo.time,
            end=None,
            velocity=None,
            pitch=int(tempo.tempo),
            Type=-1))
    tempo_items.sort(key=lambda x: x.start)

    # expand to all beat
    max_tick = tempo_items[-1].start
    existing_ticks = {item.start: item.pitch for item in tempo_items}
    wanted_ticks = np.arange(0, max_tick+1, DEFAULT_RESOLUTION)
    output = []
    for tick in wanted_ticks:
        if tick in existing_ticks:
            output.append(Item(
                name='Tempo',
                start=tick,
                end=None,
                velocity=None,
                pitch=existing_ticks[tick],
                Type=-1))
        else:
            output.append(Item(
                name='Tempo',
                start=tick,
                end=None,
                velocity=None,
                pitch=output[-1].pitch,
                Type=-1))
    tempo_items = output
    return note_items, tempo_items


class Event(object):
    def __init__(self, name, time, value, text, Type):
        self.name = name
        self.time = time
        self.value = value
        self.text = text
        self.Type = Type

    def __repr__(self):
        return 'Event(name={}, time={}, value={}, text={}, Type={})'.format(
            self.name, self.time, self.value, self.text, self.Type)


def item2event(groups, task):
    # [Bar, Position, Pitch, Velocity, Duration, Tempo]
    events = []
    n_downbeat = 0
    assert groups[0][1].name == "Tempo"
    tempo = groups[0][1].pitch
    
    for i in range(len(groups)):
        bar_st, bar_et = groups[i][0], groups[i][-1]
        n_downbeat += 1
        new_bar = True
        
        for item in groups[i][1:-1]:
            note_tuple = []
            if item.name == "Tempo":
                tempo = item.pitch
                continue

            # Bar
            if new_bar:
                BarValue = 'New' 
                new_bar = False
            else:
                BarValue = "Continue"
            note_tuple.append(Event(
                name='Bar',
                time=None, 
                value=BarValue,
                text='{}'.format(n_downbeat),
                Type=-1))

            # Position
            flags = np.linspace(bar_st, bar_et, DEFAULT_FRACTION, endpoint=False)
            index = np.argmin(abs(flags-item.start))
            note_tuple.append(Event(
                name='Position', 
                time=item.start,
                value='{}/{}'.format(index+1, DEFAULT_FRACTION),
                text='{}'.format(item.start),
                Type=-1))
            
            # Pitch
            velocity_label = np.searchsorted(LABEL_VELOCITY_BINS, item.velocity, side='right') - 1
            # store task label in pitchType
            if task == 'melody':
                pitchType = item.Type
            elif task == 'velocity':
                pitchType = velocity_label
            else:
                pitchType = -1
                
            note_tuple.append(Event(
                name='Pitch',
                time=item.start, 
                value=item.pitch,
                text='{}'.format(item.pitch),
                Type=pitchType))

            # Velocity (interval = 2)
            velocity_ind = DEFAULT_VELOCITY_BINS[np.argmin(abs(DEFAULT_VELOCITY_BINS-item.velocity))]
            note_tuple.append(Event(
                name='Velocity',
                time=item.start, 
                value=velocity_ind,
                text='{}'.format(velocity_ind),
                Type=-1))

            # Duration
            duration = item.end - item.start
            index = np.argmin(abs(DEFAULT_DURATION_BINS-duration))
            note_tuple.append(Event(
                name='Duration',
                time=item.start,
                value=index,
                text='{}/{}'.format(duration, DEFAULT_DURATION_BINS[index]),
                Type=-1))

            # Tempo
            tempo_grid = DEFAULT_TEMPO_BINS[np.argmin(abs(DEFAULT_TEMPO_BINS-tempo))]
            note_tuple.append(Event(
                name='Tempo',
                time=item.start, 
                value=tempo_grid,
                text='{}'.format(tempo_grid),
                Type=-1))
            
            events.append(note_tuple)

    return events

def item2event_remi(groups, task):
    events = []
    n_downbeat = 0
    assert groups[0][1].name == 'Tempo'

    for i in range(len(groups)):
        if 'Note' not in [item.name for item in groups[i][1:-1]]:
            continue

        bar_st, bar_et = groups[i][0], groups[i][-1]
        n_downbeat += 1
        
        events.append(Event(
            name='Bar',
            time=None, 
            value=None,
            text='{}'.format(n_downbeat),
            Type=-1))
        
        for item in groups[i][1:-1]:
            #position
            flags = np.linspace(bar_st, bar_et, DEFAULT_FRACTION, endpoint=False)
            index = np.argmin(abs(flags-item.start))
            events.append(Event(
                name='Position',
                time=item.start,
                value='{}/{}'.format(index+1, DEFAULT_FRACTION),
                text='{}'.format(item.start),
                Type=-1))
            if item.name == 'Note':
                # for store task label in pitchType for token-level classification
                velocity_label = np.searchsorted(LABEL_VELOCITY_BINS, item.velocity, side='right') - 1
                if task == 'melody':
                    pitchType = item.Type
                elif task == 'velocity':
                    pitchType = velocity_label
                else:
                    pitchType = -1
                events.append(Event(
                    name='Pitch',
                    time=item.start, 
                    value=item.pitch,
                    text='{}'.format(item.pitch),
                    Type=pitchType))
                
                velocity_ind = DEFAULT_VELOCITY_BINS[np.argmin(abs(DEFAULT_VELOCITY_BINS-item.velocity))]
                events.append(Event(
                    name='Velocity',
                    time=item.start, 
                    value=velocity_ind,
                    text='{}'.format(velocity_ind),
                    Type=-1))

                duration = item.end - item.start
                index = np.argmin(abs(DEFAULT_DURATION_BINS-duration))
                events.append(Event(
                    name='Duration',
                    time=item.start,
                    value=index,
                    text='{}/{}'.format(duration, DEFAULT_DURATION_BINS[index]),
                    Type=-1))
            elif item.name == 'Tempo':
                tempo = item.pitch
                tempo_grid = DEFAULT_TEMPO_BINS[np.argmin(abs(DEFAULT_TEMPO_BINS-tempo))]
                events.append(Event(
                    name='Tempo',
                    time=item.start, 
                    value=tempo_grid,
                    text='{}'.format(tempo_grid),
                    Type=-1))
    return events

def quantize_items(items, ticks=120):
    grids = np.arange(0, items[-1].start, ticks, dtype=int)
    # process
    for item in items:
        index = np.argmin(abs(grids - item.start))
        shift = grids[index] - item.start
        item.start += shift
        item.end += shift
    return items      


def group_items(items, max_time, ticks_per_bar=DEFAULT_RESOLUTION*4):
    items.sort(key=lambda x: x.start)
    downbeats = np.arange(0, max_time+ticks_per_bar, ticks_per_bar)
    groups = []
    for db1, db2 in zip(downbeats[:-1], downbeats[1:]):
        insiders = []
        for item in items:
            if (item.start >= db1) and (item.start < db2):
                insiders.append(item)
        overall = [db1] + insiders + [db2]
        groups.append(overall)
    return groups
