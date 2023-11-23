export PYTHONPATH="."

ckpt_path="/home/wazenmai/Warehouse/music/midibert_checkpoint/pretrain_model.ckpt"

# melody, its output folder name will be {task}_{name}
python3 MidiBERT/finetune.py --repr=CP --task=melody --tag=best --ckpt=$ckpt_path
# velocity
# python3 MidiBERT/finetune.py --task=velociy --name=default

# composer
# python3 MidiBERT/finetune.py --task=composer --name=default

# emotion
# python3 MidiBERT/finetune.py --task=emotion --name=default

