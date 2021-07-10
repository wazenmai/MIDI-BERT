# MidiBERT-Piano

## Installation

* Python3
* ```pip install -r requirements.txt```
* [TODO] add a requirement file!

## A. Prepare Data

### 1. download dataset and preprocess

* [Pop17K](https://github.com/YatingMusic/compound-word-transformer)
* [ASAP](https://github.com/fosfrancesco/asap-dataset)
  * preprocess to have 65 pieces in qualified 4/4 time signature
* [POP909](https://github.com/music-x-lab/POP909-Dataset)
  * preprocess to have 865 pieces in qualified 4/4 time signature
  * ```exploratory.py``` to get pieces qualified in 4/4 time signature and save at ```qual_pieces.pkl```
  * ```preprocess.py``` to realign and preprocess
  * Special thanks to Shih-Lun (Sean) Wu
* [Pianist8](https://zenodo.org/record/5089279)
* [EMOPIA](https://annahung31.github.io/EMOPIA/)

### 2. prepare dict

```dict/make_dict.py``` customize the events & words you'd like to add.

In this paper, we only use *Bar*, *Position*, *Pitch*, *Duration*.  And we provide our dictionaries in CP & REMI representation.

```dict/CP.pkl```

```dict/remi.pkl```

### 3. prepare CP & REMI

```./prepare_data/CP```

* For POP909 & Pop17K, run ```python3 POP2cp_task.py ```.  Please specify the dataset and whether you wanna prepare an answer array for a note-level task (i.e. melody extraction and velocity prediction).
* For example, ```python3 POP2cp_task.py --dataset=pop909 --task=melody --dir=[YOUR_DIR_TO_STORE_DATA (default: CP_data)]```

* [TODO] emopia, pianist8, asap dataset

```./prepare_data/remi/```

* The same logic applies to preparing REMI data. 

Acknowledgement: [CP repo](https://github.com/YatingMusic/compound-word-transformer), [remi repo](https://github.com/YatingMusic/remi/tree/6d407258fa5828600a5474354862353ef4e4e8ae)

## B. Pre-train a MidiBERT-Piano

Acknowledgement: [Huggingface](https://github.com/huggingface/transformers)

===TODO===

```./BERT/CP``` and ```./BERT/remi```

* ```main.py```
* For example, ```python3 main.py```



## C. Fine-tune & Evaluate on Downstream Tasks

```./BERT/CP``` and ```./BERT/remi```

===TODO===

* ```finetune.py```
* For instance, ```python3 finetune.py --task=melody --name=mlm_batch12```
* ```eval.py```
* For example, ```python3 eval.py --task=melody --off_cuda --ckpt=[ckpt_path]```
* Test loss & accuracy will be printed, and a confusion matrix will be saved.

## D. Baseline Model (Bi-LSTM)

```./baseline```

===TODO===

## E. Skyline

Run ```python3 cal_acc.py``` to get the accuracy  on pop909 using skyline algorithm.

Since Pop909 contains *melody*, *bridge*, *accompaniment*, yet skyline cannot distinguish  between melody and bridge.

There are 2 way to report its accuracy:

1. Consider *Bridge* as *Accompaniment*, attains 78.54% accuracy
2. Consider *Bridge* as *Melody*, attains 79.51%



## Citation

If you find this useful, please cite our paper.

```
@article{
	title={MidiBERT-Piano: Large-scale Pre-training forSymbolic Music Understanding},
	author={},
	year={},
}
```

