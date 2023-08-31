import argparse
import os
import time
import sys
import random

import numpy as np
import torch

import pytorch_lightning as pl
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks import ModelCheckpoint

from torchvision import transforms

from models.xraymodels import Resnet50

from datasets.mura import MuraDataModule

parser = argparse.ArgumentParser(description="Training on x-rays")

#########################
#### data parameters ####
#########################
parser.add_argument("--mura_data_dir", type=str, default='./input/mura-v11',
                    help="path to MURA repository")
parser.add_argument('--cast_dataset', dest='cast_dataset', action='store_true')
parser.set_defaults(cast_dataset=False)
parser.add_argument('--val_fold', type=int, default=0,
                    help="fold to use as validation set, the other folds will be used for training")


#########################
#### optim parameters ###
#########################
parser.add_argument("--epochs", default=20, type=int,
                    help="number of total epochs to run")
parser.add_argument("--batch_size", default=16, type=int,
                    help="batch size per gpu, i.e. how many unique instances per gpu")

#########################
#### pretrain params ####
#########################
parser.add_argument('--imagenet', dest='imagenet_pretrain', action='store_true',
                    help="use model pretrain on imagenet, default")
parser.add_argument('--noimagenet', dest='imagenet_pretrain', action='store_false',
                    help="don't use model pretrain on imagenet")
parser.set_defaults(imagenet_pretrain=True)

parser.add_argument("--encoder_finetune", default="", type=str,
                    help="pretrain encoder to finetune")


#########################
#### other parameters ###
#########################
parser.add_argument("--workers", default=8, type=int,
                    help="number of data loading workers")
parser.add_argument("--nodes", default=1, type=int,
                    help="number of nodes")
parser.add_argument("--checkpoint_name", default="", type=str,
                    help="name to append to checkpoint")
parser.add_argument("--checkpoint_freq", type=int, default=25,
                    help="Save the model periodically")
parser.add_argument("--seed", type=int, default=31, help="seed. If value set 0 = random seed")

parser.add_argument("--scheduler", type=str, default='CosineAnnealingLR',
                    help="CosineAnnealingLR or ReduceLROnPlateau")

parser.add_argument('--color_invert', dest='color_invert', action='store_true')
parser.set_defaults(color_invert=False)


def main():
    global args
    args = parser.parse_args()

    # set seeds
    seed = args.seed
    # if args.seed is zero, select a random seed
    if args.seed == 0:
        seed = np.random.randint(4294967295)

    # Seed everything
    pl.seed_everything(seed)

    # logger
    log_name = args.checkpoint_name + time.strftime("%Y%m%d-%H%M%S") + "_fold" + str(args.val_fold)
    logger = TensorBoardLogger('lightning_logs', name=log_name)

    # checkpoint
    checkpoint_callback = ModelCheckpoint(
        monitor='val_auroc',
        mode='max',
        save_top_k=-1,
        every_n_epochs=5,
        dirpath='checkpoints',
        filename=log_name + '-{epoch:02d}-{val_auroc:.2f}'
    )

    # set hyperparamters
    input_height = 320

    train_transform = transforms.Compose([transforms.ToTensor(),
                                          transforms.Resize((input_height, input_height), antialias=True),
                                          transforms.RandomRotation(15)])

    if args.color_invert:
        train_transform.transforms.append(ColorInvert())

    valid_transform = transforms.Compose([transforms.ToTensor(),
                                          transforms.Resize((input_height, input_height), antialias=True),
                                          ])

    muradm = MuraDataModule(args.mura_data_dir, batch_size=args.batch_size,
                                num_workers=args.workers,
                                train_transforms=train_transform,
                                test_transforms=valid_transform,
                                study_level=False,
                                cast_dataset=args.cast_dataset,
                                val_fold=args.val_fold)
    muradm.setup()

    # set encoder for finetune, default None
    encoder = None

    if args.encoder_finetune:
        print("Loading model for finetune (warning, not compatible with ssl finetune")
        pretrain_model = Resnet50C.load_from_checkpoint(args.encoder_finetune)
        encoder = pretrain_model.encoder

    model = Resnet50(out_features=1,
                  encoder=encoder,
                  imagenet_pretrain=args.imagenet_pretrain,
                  pos_weight=muradm.get_pos_weight(),
                  input_height=input_height,
                  batch_size=args.batch_size,
                  max_epochs=args.epochs,
                  scheduler=args.scheduler)


    # train model
    trainer = pl.Trainer(num_nodes=args.nodes,
                         accelerator='gpu',
                         precision='16-mixed',
                         max_epochs=args.epochs,
                         logger=logger,
                         callbacks=[checkpoint_callback],
                         num_sanity_val_steps=2,
                         #limit_train_batches=0.002,
                         #limit_val_batches=0.01,
                         )

    trainer.fit(model, datamodule=muradm)


if __name__ == "__main__":
    main()
