import argparse
import os
import time
import sys

from tqdm import tqdm

import numpy as np
import pandas as pd
import torch

from torchvision import transforms

import pytorch_lightning as pl
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks import ModelCheckpoint

from models.xraymodels import Resnet50
from models.utils import groupMax, groupMean

from datasets.mura import MuraDataModule



parser = argparse.ArgumentParser(description="Test model on x-rays")

#########################
#### data parameters ####
#########################
parser.add_argument("--mura_data_dir", type=str, default='./input/mura-v11',
                    help="path to MURA repository")
#########################
##### ckpt params  ######
#########################
parser.add_argument("--checkpoint", type=str, default="",
                    help="Checkpoint to use")

#########################
#### output params  #####
#########################
parser.add_argument("--output", type=str, default="",
                    help="output csv name")

#########################
#### other parameters ###
#########################
parser.add_argument("--batch_size", default=8, type=int,
                    help="batch size per gpu, i.e. how many unique instances per gpu")
parser.add_argument("--workers", default=10, type=int,
                    help="number of data loading workers")
parser.add_argument("--nodes", default=1, type=int,
                    help="number of nodes")
parser.add_argument("--gpus", default=1, type=int,
                    help="number of gpus per node")


def main():
    global args
    args = parser.parse_args()

    # logger
    log_name = args.checkpoint + "_test"
    logger = TensorBoardLogger('lightning_logs', name=log_name)


    lr = 2.5e-5
    weight_decay = 2e-4
    input_height = 320
    model = Resnet50
    grouped = False


    # load model or ensemble of models
    if args.checkpoint.split('.')[-1] == "ckpt":
        path = args.checkpoint
        models = [model.load_from_checkpoint(path)]
    else:
        checkpoint_dir = os.path.dirname(args.checkpoint)
        checkpoint_prefix = os.path.split(args.checkpoint)
        checkpoint_list = []
        for file in os.listdir(checkpoint_dir):
            if file.startswith(checkpoint_prefix):
                checkpoint_list.append(os.path.join(checkpoint_dir, file))
        models = []
        for checkpoint in checkpoint_list:
            models.append(model.load_from_checkpoint(checkpoint))

    # set resize height
    input_height = 320

    transform = transforms.Compose([transforms.ToTensor(),
                                          transforms.Resize((input_height, input_height), antialias=True),
                                          ])


    muradm = MuraDataModule(args.mura_data_dir, batch_size=args.batch_size,
                                num_workers=args.workers,
                                train_transforms=transform,
                                test_transforms=transform,
                                study_level=grouped, study_level_validation=True)
    muradm.setup()

    test_dataloader = muradm.test_dataloader()

    trainer = pl.Trainer(num_nodes=args.nodes, accelerator='gpu', precision=16,
                         logger=logger)

    # test models
    result_list_max = []
    result_list_mean = []
    for i, model in enumerate(models):
        print('Model: ', str(i))
        trainer.test(model, datamodule=muradm)
        if grouped:
            result_list_max.append(model.test_results['output'])
        else:
            result_list_mean.append(model.test_results['mean_output'])
            result_list_max.append(model.test_results['max_output'])

    print('saving to file: ', str(args.output))

    if args.output == "":
        # Use the checkpoint file name as output
        args.output = args.checkpoint.split('/')[-1].split('.')[0]

    ensemble_results_max = np.stack(result_list_max, axis=1).squeeze()
    print(ensemble_results_max.shape)
    df_max = pd.DataFrame(ensemble_results_max)
    df_max.to_csv(args.output + '_max.csv')
#    df_result_max = pd.DataFrame(result_list_max)
#    df_result_max.to_csv(args.output + '_result_max.csv')

    if not grouped:
        ensemble_results_mean = np.stack(result_list_mean, axis=1).squeeze()
        df_mean = pd.DataFrame(ensemble_results_mean)
        df_mean.to_csv(args.output + '_mean.csv')
#        df_result_mean = pd.DataFrame(result_list_mean)
#        df_result_mean.to_csv(args.output + '_result_mean.csv')


if __name__ == "__main__":
    main()
