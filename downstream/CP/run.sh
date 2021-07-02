export CUDA_VISIBLE_DEVICES=0
#python3 main.py -t melody --finetune --name=bertfinal
python3 eval.py -t melody --finetune --ckpt=result/melody-finetune/bertfinal/LSTM-melody-classification.pth 
