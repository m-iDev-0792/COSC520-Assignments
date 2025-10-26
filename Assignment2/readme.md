# Assignment 2 for COSC 520

Before you run any scripts in this repo, please make sure your current working directory is *Assignment2*. Also, please install all necessary packages by executing the following command in your terminal:

```
pip install -r requirements.txt
```

## Dataset Generation

A dataset generation Python script is provided to generate an arbitrary number of datasets.

For convenience, you can simply double-click one of the batch scripts to generate a dataset according to your platform:

```
Windows ===> dataset_gen.bat
Linux   ===> dataset_gen.sh
MacOS   ===> dataset_gen.command
```

If you can't execute one of these scripts, please make sure these scripts are executable on your system.

For example, you can do this on Linux/MacOS:

```
chmod +x dataset_gen*
```

## Unit Test

In this repo, I also provided a unit test script to verify the correctness of the KD tree.

```
pytest -v test_kdtree2d.py
```

Running the unit test requires pytest to be installed. So make sure you have all packages in requirements.txt installed.

## Visualization

I provided a Python script to visualize the step-by-step searching process of the KD tree. You can check it by running visualize_kd_tree_2d.py:

```
python visualize_kd_tree_2d.py
```

## Benchmark

To test the performance of the implemented KD tree algorithm under different data scales, please run kd_tree_2d.py, which compares build time, query time, and memory usage under different data numbers and optimization settings (dimension selection methods and tree construction strategy).

```
python kd_tree_2d.py
```