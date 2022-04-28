# Data Creation

All data in CP token are already in `Data/CP_data`, including the train, valid, test split.

You can also preprocess as below.

## 1. Download Dataset and Preprocess
Save the following dataset in `Dataset/`
* [Pop1K7](https://github.com/YatingMusic/compound-word-transformer)
* [ASAP](https://github.com/fosfrancesco/asap-dataset)
  * Download ASAP dataset from the link
* [POP909](https://github.com/music-x-lab/POP909-Dataset)
  * preprocess to have 865 pieces in qualified 4/4 time signature
  * ```cd data_creation/preprocess_pop909```
  * ```exploratory.py``` to get pieces qualified in 4/4 time signature and save them at ```qual_pieces.pkl```
  * ```preprocess.py``` to realign and preprocess
  * Special thanks to Shih-Lun (Sean) Wu
* [Pianist8](https://zenodo.org/record/5089279)
  * Step 1: Download Pianist8 dataset from the link
  * Step 2: Run `python3 pianist8.py` to split data by `Dataset/pianist8_(split).pkl`
* [EMOPIA](https://annahung31.github.io/EMOPIA/)
  * Step 1: Download Emopia dataset from the link
  * Step 2: Run `python3 emopia.py` to split data by `Dataset/emopia_(split).pkl`

## 2. Prepare Dictionary

```cd data_creation/prepare_data/dict/```
Run ```python make_dict.py```  to customize the events & words you'd like to add.

In this paper, we only use *Bar*, *Position*, *Pitch*, *Duration*.  And we provide our dictionaries in CP representation. (```data_creation/prepare_data/dict/CP.pkl```)

## 3. Prepare CP
Note that the CP tokens here only contain Bar, Position, Pitch, and Duration.  Please look into the repos below if you prefer the original definition of CP tokens.

All the commands are in ```scripts/prepare_data.sh```. You can directly edit the script and run it.

(Note that `export PYTHONPATH='.'` is necessary.)

### Melody task
```
python3 data_creation/prepare_data/main.py --dataset=pop909 --task=melody
```

### Velocity task
```
python3 data_creation/prepare_data/main.py --dataset=pop909 --task=velocity
```

### Composer task
```
python3 data_creation/prepare_data/main.py --dataset=pianist8 --task=composer
```

### Emotion task
```
python3 data_creation/prepare_data/main.py --dataset=emopia --task=emotion
```

### Custom input path
* A directory to many midi files
```
python3 data_creation/prepare_data/main.py --input_dir=$input_dir
```

* A single midi file
```
python3 data_creation/prepare_data/main.py --input_file="${input_dir}/pop.mid"
```

You can also specify the filename by adding `--name={name}`.

The CP tokens will be saved in ```Data/CP_data/```

Acknowledgement: [CP repo](https://github.com/YatingMusic/compound-word-transformer)
