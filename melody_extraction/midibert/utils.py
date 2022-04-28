import numpy as np
import miditoolkit
import copy

# parameters for input
DEFAULT_VELOCITY_BINS = np.linspace(0, 128, 32+1, dtype=np.int)
DEFAULT_FRACTION = 16
DEFAULT_DURATION_BINS = np.arange(60, 3841, 60, dtype=int)
DEFAULT_TEMPO_INTERVALS = [range(30, 90), range(90, 150), range(150, 210)]

# parameters for output
DEFAULT_RESOLUTION = 480

# define "Item" for general storage
class Item(object):
    def __init__(self, name, start, end, velocity, pitch):
        self.name = name
        self.start = start
        self.end = end
        self.velocity = velocity
        self.pitch = pitch

    def __repr__(self):
        return 'Item(name={}, start={}, end={}, velocity={}, pitch={})'.format(
            self.name, self.start, self.end, self.velocity, self.pitch)

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
                pitch=note.pitch))
                
    note_items.sort(key=lambda x: x.start)
    
    # tempo
    tempo_items = []
    for tempo in midi_obj.tempo_changes:
        tempo_items.append(Item(
            name='Tempo',
            start=tempo.time,
            end=None,
            velocity=None,
            pitch=int(tempo.tempo)))

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
                pitch=existing_ticks[tick]))
        else:
            output.append(Item(
                name='Tempo',
                start=tick,
                end=None,
                velocity=None,
                pitch=output[-1].pitch))
    tempo_items = output
    return note_items, tempo_items


class Event(object):
    def __init__(self, name, time, value, text):
        self.name = name
        self.time = time
        self.value = value
        self.text = text

    def __repr__(self):
        return 'Event(name={}, time={}, value={}, text={})'.format(
            self.name, self.time, self.value, self.text)


def item2event(groups):
    events = []
    n_downbeat = 0
    for i in range(len(groups)):
        if 'Note' not in [item.name for item in groups[i][1:-1]]:
            continue
        bar_st, bar_et = groups[i][0], groups[i][-1]
        n_downbeat += 1
        new_bar = True
        events.append([Event(
            name='Bar',
            time=None, 
            value=None,
            text='{}'.format(n_downbeat))])

        for item in groups[i][1:-1]:
            if item.name == 'Note':
                note_tuple = []
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
                    text='{}'.format(n_downbeat)))

                # Position
                flags = np.linspace(bar_st, bar_et, DEFAULT_FRACTION, endpoint=False)
                index = np.argmin(abs(flags-item.start))
                note_tuple.append(Event(
                    name='Position', 
                    time=item.start,
                    value='{}/{}'.format(index+1, DEFAULT_FRACTION),
                    text='{}'.format(item.start)))
                
                # Pitch
                note_tuple.append(Event(
                    name='Pitch',
                    time=item.start, 
                    value=item.pitch,
                    text='{}'.format(item.pitch)))

                # Duration
                duration = item.end - item.start
                index = np.argmin(abs(DEFAULT_DURATION_BINS-duration))
                note_tuple.append(Event(
                    name='Duration',
                    time=item.start,
                    value=index,
                    text='{}/{}'.format(duration, DEFAULT_DURATION_BINS[index])))
                
                # Velocity
                velocity_index = np.searchsorted(DEFAULT_VELOCITY_BINS, item.velocity, side='right') - 1
                note_tuple.append(Event(
                    name='Velocity',
                    time=item.start,
                    value=velocity_index,
                    text='{}/{}'.format(item.velocity, DEFAULT_VELOCITY_BINS[velocity_index])))

                events.append(note_tuple)

            elif item.name == 'Tempo':
                # Position
                flags = np.linspace(bar_st, bar_et, DEFAULT_FRACTION, endpoint=False)
                index = np.argmin(abs(flags-item.start))
                position = Event(
                    name='Position', 
                    time=item.start,
                    value='{}/{}'.format(index+1, DEFAULT_FRACTION),
                    text='{}'.format(item.start))
                # tempo
                tempo = item.pitch
                if tempo in DEFAULT_TEMPO_INTERVALS[0]:
                    tempo_style = Event('Tempo Class', item.start, 'slow', None)
                    tempo_value = Event('Tempo Value', item.start,
                        tempo-DEFAULT_TEMPO_INTERVALS[0].start, None)
                elif tempo in DEFAULT_TEMPO_INTERVALS[1]:
                    tempo_style = Event('Tempo Class', item.start, 'mid', None)
                    tempo_value = Event('Tempo Value', item.start,
                        tempo-DEFAULT_TEMPO_INTERVALS[1].start, None)
                elif tempo in DEFAULT_TEMPO_INTERVALS[2]:
                    tempo_style = Event('Tempo Class', item.start, 'fast', None)
                    tempo_value = Event('Tempo Value', item.start,
                        tempo-DEFAULT_TEMPO_INTERVALS[2].start, None)
                elif tempo < DEFAULT_TEMPO_INTERVALS[0].start:
                    tempo_style = Event('Tempo Class', item.start, 'slow', None)
                    tempo_value = Event('Tempo Value', item.start, 0, None)
                elif tempo > DEFAULT_TEMPO_INTERVALS[2].stop:
                    tempo_style = Event('Tempo Class', item.start, 'fast', None)
                    tempo_value = Event('Tempo Value', item.start, 59, None)

                tempo_tuple = [position, tempo_style, tempo_value]
                events.append(tempo_tuple)

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
