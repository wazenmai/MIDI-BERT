# Generate Pianoroll

The pianoroll diagram can give us a more intuitive understanding of each melody-extraction method.

## Ground Truth
Use following code to generate CP data for ground truth and MidiBERT.
```python
import numpy as np
from data_creation.prepare_data.model import *

dir_path = "melody_extraction/pianoroll/audio/" # modify to your own directory path
song_path = os.path.join(dir_path, "018.mid") # modify the filename
output_name = "pop909_018" # modify the output name

songs = [song_path]
model = CP(dict="data_creation/prepare_data/dict/CP.pkl")
word, ys = model.prepare_data(songs, 'melody', 512)
np.save(os.path.join(dir_path, output_name + "_cp.npy"), word)
np.save(os.path.join(dir_path, output_name + "_groundtruth.npy"), ys)
```

Set the data path and mode in `settings.py`, and run the script from the main directory of MidiBERT by `bash scripts/pianoroll.sh`.

## MidiBERT-CP
1. Get the prediction by running `MidiBERT/eval.py` by `scripts/eval.sh`. You need to modify the code to change the test data path and save the prediction.
2. Run the script `bash scripts/pianoroll.sh` and make sure the settings is correct. 

## Skyline
### Skyline from Midi
Simply uncomment the code in `plot.py` and modify the midi file's name.
```python
sky_melody, sky_accomp = skyline_from_midi("018.mid")
plot_roll([sky_melody, sky_accomp], 48*4*4)
```
### Skyline from CP
In order to get the fair comparison to our MidiBERT, we use the CP data to align the skyline generated melody. After generating the CP data, set the mode to `skyline` and run the script `bash scripts/pianoroll.sh`.

