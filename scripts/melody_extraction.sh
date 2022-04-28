song_path="resources/Adele.mid"
model_path="result/finetune/melody_default/model_best.ckpt"

export PYTHONPATH='.'
python3 melody_extraction/midibert/extract.py --input=$song_path --ckpt=$model_path --cpu --bridge=False

