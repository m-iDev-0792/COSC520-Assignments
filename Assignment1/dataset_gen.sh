mkdir -p dataset
mkdir -p dataset_false
python dataset_gen.py 100000100 -o dataset
python dataset_gen.py 100000100 -o dataset_false