# MidiBERT

## 1. Pre-train a MidiBERT-Piano
### Pre-train (default)
```./scripts/pretrain.sh```
You can change the output directory name by specifying `--name={name}`.

A folder named ```result/pretrain/{name}/``` will be created, with checkpoint & log inside.

### Customize your own pre-training dataset

Feel free to select given dataset and add your own dataset.  To do this, add ```--dataset```, and specify the respective path in ```load_data()``` function.

For example,
```python
# To pre-train a model with only 2 datasets
export PYTHONPATH='.'
python3 main.py --name=default --dataset pop1k7 asap	
``` 

Acknowledgement: [HuggingFace](https://github.com/huggingface/transformers), [codertimo/BERT-pytorch](https://github.com/codertimo/BERT-pytorch)

Special thanks to Chin-Jui Chang

## 2. Fine-tune on Downstream Tasks
```./scripts/finetune.sh```

A folder named ```result/finetune/{name}/``` will be created, with checkpoint & log inside.

## 3. Evaluation
```./scripts/eval.sh```

```python
python3 eval.py --task=melody --cpu --ckpt=[ckpt_path]
```

Test loss & accuracy will be printed, and a figure of confusion matrix will be saved in the same directory as the checkpoint.
