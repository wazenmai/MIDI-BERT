import pickle

event2word = {'Bar': {}, 'Position': {}, 'Pitch': {}, 'Duration': {}, 'Velocity': {}, 'Tempo': {}, 'Chord': {}, 'Type': {}}
word2event = {'Bar': {}, 'Position': {}, 'Pitch': {}, 'Duration': {}, 'Velocity': {}, 'Tempo': {}, 'Chord': {}, 'Type': {}}

def special_tok(cnt, cls):
    '''event2word[cls][cls+' <SOS>'] = cnt
    word2event[cls][cnt] = cls+' <SOS>'
    cnt += 1

    event2word[cls][cls+' <EOS>'] = cnt
    word2event[cls][cnt] = cls+' <EOS>'
    cnt += 1'''

    event2word[cls][cls+' <PAD>'] = cnt
    word2event[cls][cnt] = cls+' <PAD>'
    cnt += 1

    event2word[cls][cls+' <MASK>'] = cnt
    word2event[cls][cnt] = cls+' <MASK>'
    cnt += 1


# Bar
cnt, cls = 0, 'Bar'
event2word[cls]['Bar New'] = cnt
word2event[cls][cnt] = 'Bar New'
cnt += 1

event2word[cls]['Bar Continue'] = cnt
word2event[cls][cnt] = 'Bar Continue'
cnt += 1
special_tok(cnt, cls)

# Position
cnt, cls = 0, 'Position'
for i in range(1, 17):
    event2word[cls][f'Position {i}/16'] = cnt
    word2event[cls][cnt]= f'Position {i}/16'
    cnt += 1

special_tok(cnt, cls)

# Note On
cnt, cls = 0, 'Pitch'
for i in range(22, 108):
    event2word[cls][f'Pitch {i}'] = cnt
    word2event[cls][cnt] = f'Pitch {i}'
    cnt += 1

special_tok(cnt, cls)

# Note Duration
cnt, cls = 0, 'Duration'
for i in range(64):
    event2word[cls][f'Duration {i}'] = cnt
    word2event[cls][cnt] = f'Duration {i}'
    cnt += 1

special_tok(cnt, cls)

# Note Velocity
cnt, cls = 0, 'Velocity'
event2word[cls][0] = 0
word2event[cls][0] = 0
cnt += 1
for i in range(40, 69):
    event2word[cls][f'Velocity {i}'] = cnt
    word2event[cls][cnt] = f'Velocity {i}'
    cnt += 1

special_tok(cnt, cls)


# Tempo
cnt, cls = 0, 'Tempo'
event2word[cls][0] = 0
word2event[cls][0] = 0
event2word[cls]["CONTI"] = 1
word2event[cls][1] = "CONTI"
cnt += 2
for i in range(32, 225, 3):
    event2word[cls][f'Tempo {i}'] = cnt
    word2event[cls][cnt] = f'Tempo {i}'
    cnt += 1
special_tok(cnt, cls)

# Chord
cnt, cls = 0, 'Chord'
event2word[cls][0] = 0
word2event[cls][0] = 0
cnt += 1
note = {'A', 'A#', 'B', 'C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#'}
chord_type = {'+', '/o7', 'M', 'M7', 'm', 'm7', 'o', 'o7', 'sus2', 'sus4'}
for n in note:
    for c in chord_type:
        event2word[cls][f'{n}_{c}'] = cnt
        word2event[cls][cnt] = f'{n}_{c}'
        cnt += 1
special_tok(cnt, cls)

# Type
cnt, cls = 0, 'Type'
event2word[cls]['EOS'] = cnt
word2event[cls][cnt] = 'EOS'
cnt += 1
event2word[cls]['Metrical'] = cnt
word2event[cls][cnt] = 'Metrical'
cnt += 1
event2word[cls]['Note'] = cnt
word2event[cls][cnt] = 'Note'
cnt += 1
special_tok(cnt, cls)

# print(event2word)
# print(word2event)
event_sum = 0
for k in event2word:
    event_sum += len(event2word[k])
    print(k)
    print(event2word[k])
print("num: ", event_sum)

t = (event2word, word2event)

with open('comp_CP.pkl', 'wb') as f:
    pickle.dump(t, f)

