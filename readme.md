# MidiBERT-Piano

If you'd like to reproduce the results (MidiBERT) shown in the paper, please 

1. download the [checkpoints](https://drive.google.com/drive/folders/1ceIfC1UugZQHPgpEEMkdAF0VhZ1EeLl3?usp=sharing), and rename files like the following

```
under folder MidiBERT/CP/
result
└── finetune
	└── melody_default
		└── model_best.ckpt
	└── velocity_default
		└── model_best.ckpt
	└── composer_default
		└── model_best.ckpt
	└── emotion_default
		└── model_best.ckpt
```



2. refer to **C. 2.evaluation**.  

![image-20210710185007453](fig/result.png)

, and you are free to go!  *(btw, no gpu is needed)*



## Installation

* Python3
* ```pip install -r requirements.txt```
* [TODO] add a requirement file! (有可以自動抓package的code)

## A. Prepare Data

All data in CP/REMI token are stored in ```data/CP``` & ```data/remi```, respectively, including the train, valid, test split.

You can also preprocess as below.

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

==TODO== model_CP.py 改成 utility

```./prepare_data/CP```

* For POP909 & Pop17K, run ```python3 POP2cp_task.py ```.  Please specify the dataset and whether you wanna prepare an answer array for a note-level task (i.e. melody extraction and velocity prediction).
* For example, ```python3 POP2cp_task.py --dataset=pop909 --task=melody --dir=[DIR_TO_STORE_DATA]```

* [TODO] emopia, pianist8, asap dataset

```./prepare_data/remi/```

* The same logic applies to preparing REMI data. 

Acknowledgement: [CP repo](https://github.com/YatingMusic/compound-word-transformer), [remi repo](https://github.com/YatingMusic/remi/tree/6d407258fa5828600a5474354862353ef4e4e8ae)

You may encode these midi files in different representations, the data split is in ***.

## B. Pre-train a MidiBERT-Piano

```./BERT/CP``` and ```./BERT/remi```

* ```main.py```
* For example, ```python3 main.py --name=default```.
* A folder named ```CP_result/pretrain/default/``` will be created, with checkpoint & log inside.
* Feel free to select given dataset and add your own dataset.  To do this, add ```--dataset```, and specify the respective path in ```load_data()``` function.
* Ex: ```python3 main.py --name=default --dataset pop1k7 asap``` to pre-train a model with only 2 datasets

Acknowledgement: [HuggingFace](https://github.com/huggingface/transformers)

Special thanks to Chin-Jui Chang

## C. Fine-tune & Evaluate on Downstream Tasks

```./BERT/CP``` and ```./BERT/remi```

### 1. fine-tuning

* ```finetune.py```
* For instance, ```python3 finetune.py --task=melody --name=default```
* A folder named ```CP_result/finetune/{name}/``` will be created, with checkpoint & log inside.

### 2. evaluation

* ```eval.py```
* For example, ```python3 eval.py --task=melody --cpu --ckpt=[ckpt_path]```
* Test loss & accuracy will be printed, and a figure of confusion matrix will be saved.

The same logic applies to REMI representation. 

## D. Baseline Model (Bi-LSTM)

```./baseline```

===TODO===

Special thanks to Ching-Yu (Sunny) Chiu

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

