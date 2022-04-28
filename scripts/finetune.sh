export PYTHONPATH="."

# melody
python3 MidiBERT/finetune.py --task=melody --name=melody_default --cpu

# velocity
#python3 MidiBERT/finetune.py --task=velociy --name=velocity_default

# composer
#python3 MidiBERT/finetune.py --task=composer --name=composer_default

# emotion
#python3 MidiBERT/finetune.py --task=emotion --name=emotion_default

