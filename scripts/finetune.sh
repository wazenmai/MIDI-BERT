export PYTHONPATH="."

ckpt_path="/home/wazenmai/Warehouse/music/midibert_checkpoint/pretrain_model.ckpt"

pretrain_ckpt=MidiBERT/result/pretrain_CP/test/model_best.ckpt
#python3 MidiBERT/finetune.py --task=melody --tag=default --ckpt=$pretrain_ckpt --repr=CP

# melody, its output folder name will be {task}_{name}
python3 MidiBERT/finetune.py --repr=CP --task=melody --tag=best --ckpt=$ckpt_path
# velocity
python3 MidiBERT/finetune.py --task=velocity --tag=default --ckpt=$pretrain_ckpt --repr=CP

# composer
python3 MidiBERT/finetune.py --task=composer --tag=default --ckpt=$pretrain_ckpt --repr=CP

# emotion
python3 MidiBERT/finetune.py --task=emotion --tag=default --ckpt=$pretrain_ckpt --repr=CP

