import pickle

event2word = {'Bar': {}, 'Position': {}, 'Pitch': {}, 'Velocity': {}, 'Duration': {}, 'Tempo': {}}
word2event = {'Bar': {}, 'Position': {}, 'Pitch': {}, 'Velocity': {}, 'Duration': {}, 'Tempo': {}}

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

# Velocity
# DEFAULT_VELOCITY_BINS = np.linspace(0,  128, 64+1, dtype=np.int)
cnt, cls = 0, 'Velocity'
for i in range(0, 130, 2):
    event2word[cls][f'Velocity {i}'] = cnt
    word2event[cls][cnt] = f'Velocity {i}'
    cnt += 1

special_tok(cnt, cls)

# Note Duration
cnt, cls = 0, 'Duration'
for i in range(64):
    event2word[cls][f'Duration {i}'] = cnt
    word2event[cls][cnt] = f'Duration {i}'
    cnt += 1

special_tok(cnt, cls)

# Tempo
# DEFAULT_BPM_BINS = np.linspace(32, 224, 64+1, dtype=np.int)
cnt, cls = 0, 'Tempo'
for i in range(32, 227, 3):
    event2word[cls][f'Tempo {i}'] = cnt
    word2event[cls][cnt] = f'Tempo {i}'
    cnt += 1

special_tok(cnt, cls)


print(event2word)
print(word2event)
t = (event2word, word2event)

with open('CP.pkl', 'wb') as f:
    pickle.dump(t, f)
