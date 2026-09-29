# bone_bias

## Introduction

This repository allow to train and test a resnet50 model on the MURA dataset

You need to install the libraries from requirements.txt, `pip install -r requirements.txt` (versions are pinned, in particular numpy/pandas, to avoid binary incompatibilities).

## Training

The train.py script is used to train the neural network.
This requires the MURA dataset, the default location is './input/mura-v11' but can be changed with the --mura_data_dir argument.
A list of all parameters can be seen with the `train.py -h` command.

### Debiasing with cast crops

`--cast_dataset` adds the crops of immobilization devices (casts and splints), labeled as normal, to the training folds.
Each crop is repeated `--cast_redundancy` times (default 20, the value used in the paper) and the class weight of the loss
is recomputed on the augmented training set.

## Evaluating

The eval.py script is used to generate the predictions on the test set.
It generates two csv files with results grouped by studies, one with a max pooling of the results from the images, on with the mean pooling.

## Statistics

The stats.ipynb notebook uses the csv files from the evalution script to generate statitics about the results of the ensemble of models on the test set.
This notebook also requires the installation of the sklearn library and jupyter notebook in addition to the requirements.txt.

## Saliency maps

The resnetexpl.py script is used to generate the saliency maps by using a trained checkpoint.
This requires the installation of the PolyCAM library (https://github.com/aenglebert/polycam).

## Additional analyses (`analysis/`)

Scripts used for the statistical, calibration and saliency analyses of the paper. They only need the CSV files written by `eval.py`, the MURA validation labels and, for the saliency review, the figures written by `resnetexpl.py`. See `analysis/README.md` for details.

- `analysis/check_patient_leakage.py` verifies that the cross-validation folds and the cast crops are patient-disjoint and that no training patient is in the test set.
- `analysis/compare_models.py` compares two models on the same test set (paired DeLong test, paired patient-cluster bootstrap, exact McNemar test, non-inferiority, calibration, per-region comparison):
  `python analysis/compare_models.py --labels valid_labeled_studies.csv --base 'imagenet2023*_fold*_mean.csv' --mod 'cast20imagenet2023*_fold*_mean.csv' --out compare_results`
- `analysis/saliency_review.py` runs a blinded, paired review of the saliency maps of two models (device presence on the X-rays alone, then "does the device drive the abnormal prediction?" on shuffled, anonymised maps) and computes the exact McNemar test.
