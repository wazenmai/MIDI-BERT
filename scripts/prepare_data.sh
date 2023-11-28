data_dir="./Dataset/"
output_dir="./Data/tmp"
intput_dir="./Data/tmp"

export PYTHONPATH='.'

# melody
python3 data_creation/prepare_data/main.py --dataset=pop909 --task=melody --data_path $data_dir --output_dir $output_dir --mode CP

# velocity
python3 data_creation/prepare_data/main.py --dataset=pop909 --task=velocity

# composer
python3 data_creation/prepare_data/main.py --dataset=pianist8 --task=composer

# emotion
python3 data_creation/prepare_data/main.py --dataset=emopia --task=emotion

# custom directory
python3 data_creation/prepare_data/main.py --input_dir=$input_dir

# custom single file
python3 data_creation/prepare_data/main.py --input_file="${input_dir}/pop.mid"
