export PYTHONPATH="."

# add --cpu if you are not using GPU resources

# melody
python3 MidiBERT/eval.py --task=melody 

# velocity
python3 MidiBERT/eval.py --task=velocity 

# composer
python3 MidiBERT/eval.py --task=composer

# emotion
python3 MidiBERT/eval.py --task=emotion

