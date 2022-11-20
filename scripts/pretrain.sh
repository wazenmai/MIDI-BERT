export PYTHONPATH="."
export CUDA_VISIBLE_DEVICES=0

python3 MidiBERT/main.py --name=test --repr=CP
