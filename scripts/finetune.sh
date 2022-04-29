export PYTHONPATH="."

# melody, its output folder name will be {task}_{name}
python3 MidiBERT/finetune.py --task=melody --name=default

# velocity
python3 MidiBERT/finetune.py --task=velociy --name=default

# composer
python3 MidiBERT/finetune.py --task=composer --name=default

# emotion
python3 MidiBERT/finetune.py --task=emotion --name=default

