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


