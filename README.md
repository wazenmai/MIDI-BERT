# MIDI-BERT

## MIDI Library
- [miditoolkit](https://github.com/YatingMusic/miditoolkit)
- [pretty_midi](https://craffel.github.io/pretty-midi/)

## Tracing REMI-Transformer
- [REMI](https://github.com/YatingMusic/remi)

In `finetune.py`, it uses `model.prepare_data` to transform training data to token.

In function `prepare_data`, it uses `extract_events` to get the event from MIDI data.

Then, in `extract_event`, it uses functions in `utils.py` to read items from MIDI data, then transform it to events.

If you want to know how events transform to tokens, you could download [REMI-tempo-checkpoint](https://drive.google.com/file/d/1gxuTSkF51NP04JZgTE46Pg4KQsbHQKGo/view), use `pickle` library to read `dictionary.pkl` for details.

## Pre-train Dataset
- [MAESTRO](https://magenta.tensorflow.org/datasets/maestro)
- [ASAP](https://github.com/fosfrancesco/asap-dataset)


## Finetune Dataset

1. [Rule Mining for Local Boundary Detection in Melodies](https://program.ismir2020.net/poster_2-14.html)
    Melody segmentation seems a nice task.  There are in total three datasets with a reasonable number of boundary labels (see Table 1). 
    1. [MTC](http://www.liederenbank.nl/mtc/)
    2. [ESSEN](http://www.esac-data.org/)
    3. CHOR

2. [The Jazz Harmony Treebank](https://program.ismir2020.net/poster_2-06.html)
    Symbolic domain chord recognition!  This might be an interesting downstream task that can benefit from pre-training the token embeddings.
    - [Jazz Harmony Treebank](https://github.com/DCMLab/JazzHarmonyTreebank)

3. [Voice-leading Schema Recognition Using Rhythm and Pitch Features](https://program.ismir2020.net/poster_4-07.html)
    Their dataset is based on the full set of Mozart’s piano sonatas encoded in MusicXML format.

4. [Classifying Leitmotifs in Recordings of Operas by Richard Wagner](https://program.ismir2020.net/poster_4-01.html)
=> This paper actually concerns with audio input but it's about motifs; [website](https://www.merriam-webster.com/words-at-play/difference-between-motif-and-leitmotif)

5. Symbolic domain melody identification from a polyphonic piece
    [paper](https://archives.ismir.net/ismir2019/paper/000114.pdf)
    [code](https://github.com/LIMUNIMI/Symbolic-Melody-Identification)

## Baseline Model
- Estimating musical time information from performed MIDI files
    symbolic-domain beat/downbeat tracking
    [tool]( https://mir.sechsachtel.de/midi/)
    [paper]

## Reference
[Majenta blog posts](https://magenta.tensorflow.org/blog) 
=> there are many fun projects done by Majenta team.


