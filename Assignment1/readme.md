# COSC 520 Assignment1 - Login Checker

## Setup environment
Please do the following steps to setup the environment:
```
pip install -r requirements.txt
```

## Generate dataset
Please do the following steps to generate the dataset:
```
./dataset_gen.sh
```

## Run the unit test
Please do the following steps to run the unit test, to make sure the code is working correctly:
```
python unit_test.py
```

## Benchmark
Run the benchmark to test the performance of the login checker:
```
python benchmark.py
```
After the benchmark is finished, the results will be saved in the results_10K.json and results_10percent.json files. The line graph will be shown to compare the performance of the login checker.