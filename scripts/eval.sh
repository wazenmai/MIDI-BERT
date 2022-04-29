export PYTHONPATH="."

# melody
python3 MidiBERT/eval.py --task=melody --cpu

# velocity
python3 MidiBERT/eval.py --task=velociy --cpu

# composer
python3 MidiBERT/eval.py --task=composer --cpu

# emotion
python3 MidiBERT/eval.py --task=emotion --cpu

