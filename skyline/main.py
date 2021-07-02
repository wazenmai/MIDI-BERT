from analyzer import extract_melody
import glob
import miditoolkit.midi.parser as midparser

def extract(file, f):
    obj = midparser.MidiFile(file)
    melody = obj.instruments[0].notes
    bridge = obj.instruments[1].notes
    all_notes = []
    all_notes.extend(melody)
    all_notes.extend(bridge)
    all_notes.extend(obj.instruments[2].notes)

    all_notes.sort(key=lambda x:(x.start))

    # extract
    pred = extract_melody(all_notes)
    print(pred)
    exit(1)
    MnB = melody+bridge
    all_notes, melody, pred, bridge, MnB = set(all_notes), set(melody), set(pred), set(bridge), set(MnB)
    print('# all notes: {}, # melody: {}, # bridge: {}, # pred_melody: {}'.format(len(all_notes), len(melody), len(bridge), len(pred)))
    f.write('# all notes: {}, # melody: {}, # bridge: {}, # pred_melody: {}\n'.format(len(all_notes), len(melody), len(bridge), len(pred)))

    intersection = melody.intersection(pred)
    intersection2 = MnB.intersection(pred)
    acc, acc2 = len(intersection), len(intersection2)
    sum1 = len(melody)+len(pred)-acc
    sum2 = len(melody)+len(bridge)+len(pred)-acc2
    print('accuracy (melody only):', acc/sum1)
    print('accuracy (melody & bridge):', acc2/sum2)
    return acc, acc2, sum1, sum2

def main():
    root_dir = '/home/yh1488/NAS-189/home/Dataset/pop909_aligned/'
    files = glob.glob(root_dir+'*.mid')
    # use test set ONLY
    files = files[-86:]
    all_acc1, all_acc2, cnt1, cnt2 = 0,0,0,0

    with open('pop909.log','a') as f:
        f.write('root_dir: ' + root_dir + '\n')
        f.write('# file: ' + str(len(files)) + '\n')
        for file in files:
            print('\n[',file,']')
            f.write('\n[ ' + file.split('/')[-1] + ' ]\n')
            acc1, acc2, sum1, sum2 = extract(file, f)
            all_acc1 += acc1
            all_acc2 += acc2
            cnt1 += sum1
            cnt2 += sum2
            f.write('acc(melody only): ' + str(acc1/sum1) + '\n')
            f.write('acc(melody + bridge): ' + str(acc2/sum2) + '\n')
    
        f.write('\navg_acc (melody only): ' + str(all_acc1/cnt1) + '\n')
        f.write('avg_acc (melody & bridge): ' + str(all_acc2/cnt2))


if __name__ == '__main__':
    main()
