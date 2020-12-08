import pickle

event2word = {}
word2event = {}
cnt = 0

# Bar
event2word['Bar_None'] = cnt
word2event[cnt] = 'Bar_None'
cnt += 1

# Position
for i in range(1, 17):
    event2word[f'Position_{i}/16'] = cnt
    word2event[cnt]= f'Position_{i}/16'
    cnt += 1

# Tempo 
event2word['Tempo Class_slow'] = cnt
word2event[cnt] = 'Tempo Class_slow'
cnt += 1
event2word['Tempo Class_mid'] = cnt
word2event[cnt] = 'Tempo Class_mid'
cnt += 1
event2word['Tempo Class_fast'] = cnt
word2event[cnt] = 'Tempo Class_fast'
cnt += 1

for i in range(60):
    event2word[f'Tempo Value_{i}'] = cnt
    word2event[cnt] = f'Tempo Value_{i}'
    cnt += 1

# Note Velocity
for i in range(40):
    event2word[f'Note Velocity_{i}'] = cnt
    word2event[cnt] = f'Note Velocity_{i}'
    cnt += 1

# Note Duration
for i in range(64):
    event2word[f'Note Duration_{i}'] = cnt
    word2event[cnt] = 'Note Duration_{i}'
    cnt += 1

# Note On
for i in range(22, 108):
    event2word[f'Note On_{i}'] = cnt
    word2event[cnt] = f'Note On_{i}'
    cnt += 1

t = (event2word, word2event)

with open('bert_dict.pkl', 'wb') as f:
    pickle.dump(t, f)


print(cnt)
