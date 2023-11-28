export PYTHONPATH="."

pretrain_ckpt=MidiBERT/result/pretrain_CP/test/model_best.ckpt


# melody, its output folder name will be {task}_{name}
python3 MidiBERT/finetune.py --task=melody --tag=default --ckpt=$pretrain_ckpt --repr=CP

# velocity
python3 MidiBERT/finetune.py --task=velocity --tag=default --ckpt=$pretrain_ckpt --repr=CP

# composer
python3 MidiBERT/finetune.py --task=composer --tag=default --ckpt=$pretrain_ckpt --repr=CP

# emotion
python3 MidiBERT/finetune.py --task=emotion --tag=default --ckpt=$pretrain_ckpt --repr=CP

