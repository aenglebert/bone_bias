# bone_bias

## Introduction

This repository allow to train and test a resnet50 model on the MURA dataset

You need to install the libraries from requirements.txt, `pip install -r requirements.txt`

## Training

The train.py script is used to train the neural network.
This requires the MURA dataset, the default location is './input/mura-v11' but can be changed with the --mura_data_dir argument.
A list of all parameters can be seen with the `train.py -h` command.

## Evaluating

The eval.py script is used to generate the predictions on the test set.
It generates two csv files with results grouped by studies, one with a max pooling of the results from the images, on with the mean pooling.

## Statistics

The stats.ipynb notebook uses the csv files from the evalution script to generate statitics about the results of the ensemble of models on the test set.
This notebook also requires the installation of the sklearn library and jupyter notebook in addition to the requirements.txt.

## Saliency maps

The resnetexpl.py script is used to generate the saliency maps by using a trained checkpoint.
This requires the installation of the PolyCAM library (https://github.com/aenglebert/polycam).
