input_dir="../example_midis"

export PYTHONPATH='.'

# melody
python3 data_creation/prepare_data/main.py --dataset=pop909 --task=melody

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
