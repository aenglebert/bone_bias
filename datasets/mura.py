import os

import numpy as np
import pandas as pd

import torch
from torch.utils.data import Dataset, DataLoader, random_split

import torchvision
from torchvision import transforms

import pytorch_lightning as pl

from PIL import Image

from tqdm import tqdm

from functools import partial


def study_collate_fn(data, max_images):
    """Creates mini-batch tensors from the list of tuples (image_list, label).

    Since the image list is variable lenght we need a custom collate_fn

    Args:
        data: list of tuple (image_list, label).
            - imageList: list of image tensors.
            - label: torch tensor of shape 1.
        max_images: int
            limit to that number of images in total, discare last studies if exceed and print warning
    Returns:
        images: torch tensor of shape (total_images_nb, 3, w, h).
        seq_sizes: list of size of original image sequences sizes
        labels: tensor of stacked labels.
    """

    images = []
    labels = []
    seq_sizes = []

    max_remaining = max_images

    for image_list, label in data:
        seq_sizes.append(len(image_list))
        labels.append(label)
        if len(image_list) > max_remaining:
            print("******************************************************************\n"
                  "Batch limited to " + str(max_images) + ", some images are discared\n"
                  "******************************************************************")
            break
        for image in image_list:
            images.append(image)

    # Merge images (from list of 3D tensor to 4D tensor).
    images = torch.stack(images, 0)

    # convert labels to tensor
    labels = torch.tensor(labels)

    return images, seq_sizes, labels


def images_from_study_collate_fn(data, max_images):
    """Creates mini-batch tensors from the list of tuples (image_list, label).

    Since the image list is variable lenght we need a custom collate_fn

    Args:
        data: list of tuple (image_list, label).
            - imageList: list of image tensors.
            - label: torch tensor of shape 1.
    Returns:
        images: torch tensor of shape (total_images_nb, 3, w, h).
        labels: tensor of stacked labels (total_images_nb).
    """

    images = []
    labels = []
    seq_sizes = []

    max_remaining = max_images

    for image_list, label in data:
        if len(image_list) > max_remaining:
            print("******************************************************************\n"
                  "Batch limited to " + str(max_images) + ", some images are discared\n"
                  "******************************************************************")
            break
        for image in image_list:
            images.append(image)
            labels.append(label)

    # Merge images (from list of 3D tensor to 4D tensor).
    images = torch.stack(images, 0)

    # convert labels to tensor
    labels = torch.tensor(labels)

    return images, labels


def mura_fold_df(data_dir,
                 mura_csv_prefix,
                 cast_csv_prefix,
                 val_fold=0,
                 n_folds=5,
                 cast_redundancy=10):
    """
    Generate DataFrames for train and validation from folds
    :param data_dir: base data directory
    :param mura_csv_prefix: prefix of train csv without number and extension
    :param cast_csv_prefix: prefix of validation csv without number and extension
    :param val_fold: number of validation fold
    :param n_folds: number of folds
    :param cast_redundancy: redundancy of cast dataset
    :return: train_df, valid_df
    """
    print("Preparing MURA " + str(n_folds) + "-folds Dataframes")
    mura_csv_files = []
    cast_csv_files = []
    mura_df_list = []
    cast_df_list = []
    train_df_list = []
    valid_df_list = []
    for i in range(n_folds):
        mura_csv_files.append(os.path.join(data_dir, mura_csv_prefix + str(i) + '.csv'))
        cast_csv_files.append(os.path.join(data_dir, cast_csv_prefix + str(i) + '.csv'))
        mura_df_list.append(pd.read_csv(mura_csv_files[i], header=None))
        cast_df_list.append(pd.read_csv(cast_csv_files[i], header=None))
        if i == val_fold:
            valid_df_list.append(mura_df_list[i])
            for j in range(cast_redundancy):
                valid_df_list.append(cast_df_list[i])
        else:
            train_df_list.append(mura_df_list[i])
            for j in range(cast_redundancy):
                train_df_list.append(cast_df_list[i])
    train_df = pd.concat(train_df_list, ignore_index=True)
    valid_df = pd.concat(valid_df_list, ignore_index=True)
    return train_df, valid_df


class MURADataset(Dataset):
    def __init__(self, data_dir, studies, transform=None, study_level=False, cast_csv=None, cast_redundancy=10):
        """
        Args:
            csv_file: file with file path and labels ({train,test}_labeled_studies.csv)
            data_dir: directory containing the dataset
            transform: transformation to apply
            verbose_label: output the type of exam in a onehot vector in addition
                to the positive/negative label
            """
        if isinstance(studies, str):
            csv_file = studies
            self.df = pd.read_csv(csv_file, header=None)
            if cast_csv:
                cast_df = pd.read_csv(cast_csv, header=None)
                for i in range(cast_redundancy):
                    self.df = pd.concat([self.df, cast_df])
        elif isinstance(studies, pd.DataFrame):
            self.df = studies

        self.data_dir = data_dir

        # keep a list of list of images path, grouped by study with a label and location per study
        self.studies_path = []
        self.studies_label = []

        # Also keep a list of individuals images path with label and location of each image
        self.images_path = []
        self.images_label = []

        # loop over each study in df
        for index, row in tqdm(self.df.iterrows(), total=self.df.shape[0]):
            cur_study_images_path = []

            # study folder path
            cur_study_path = row.iloc[0]

            # study label to global list
            self.studies_label.append(row.iloc[1])

            # list images in the study folder
            cur_study_images_name = [file for file in os.listdir(os.path.join(data_dir, cur_study_path)) if
                                     not file.startswith('.')]

            # get full path for each image in the study
            for images_name in cur_study_images_name:
                images_path = os.path.join(data_dir, cur_study_path, images_name)
                # add image path to list of curent study images path
                cur_study_images_path.append(images_path)
                # also add to global list of images paths
                self.images_path.append(images_path)
                # add image label to global list
                self.images_label.append(row.iloc[1])

            # add the list of currente study images paths to global list of list
            self.studies_path.append(cur_study_images_path)

        self.studies_label = torch.tensor(self.studies_label)
        self.n_pos_studies = torch.sum(self.studies_label).item()
        self.images_label = torch.tensor(self.images_label)
        self.n_pos_images = torch.sum(self.images_label).item()

        # store parameters
        self.study_level = study_level
        self.transform = transform

    def __len__(self):
        # return number of studies or images depending of study_level
        if self.study_level:
            return len(self.studies_path)
        else:
            return len(self.images_path)

    def __pos__(self):
        # return number of positives studies or images depending of study_level
        if self.study_level:
            return self.n_pos_studies
        else:
            return self.n_pos_images

    def __getitem__(self, index):
        """
        Args:
            index: index of image or study
        Returns:
            tupple of ((list of images), label) if study_level
            tupple of (image, labels) if not study_level
        """
        if self.study_level:
            study_images = []
            # get every image in the study
            for image_path in self.studies_path[index]:
                image = Image.open(image_path).convert('RGB')

                # transform is needed
                if self.transform is not None:
                    image = self.transform(image)

                # append to list of images
                study_images.append(image)

            # get label and return with list of images
            label = self.studies_label[index]
            return study_images, label

        else:
            # get a single image
            image_path = self.images_path[index]
            image = Image.open(image_path).convert('RGB')

            # transform if needed
            if self.transform is not None:
                image = self.transform(image)

            # get image label and return with image
            label = self.images_label[index]
            return image, label


class MuraDataModule(pl.LightningDataModule):

    def __init__(self, data_dir: str = '../input/mura-v11', train_transforms=None, test_transforms=None, batch_size=32,
                 num_workers=4, val_fold=0, study_level=False, study_level_validation=False, drop_last=False,
                 cast_dataset=False):
        super().__init__()

        # set directories values
        self.data_dir = data_dir
        self.data_name = "MURA-v1.1"

        if cast_dataset is False:
            cast_redundancy = 0
            print("Don't use cast dataset")
        else:
            print("Use cast dataset")
            cast_redundancy = 5

        # get train and validation df
        self.train_df, self.valid_df = mura_fold_df(self.data_dir,
                                                    'train_fold',
                                                    'train_cast_fold',
                                                    val_fold=val_fold,
                                                    cast_redundancy=cast_redundancy)
        # import csv
        self.test_csv = os.path.join(self.data_dir, self.data_name, "valid_labeled_studies.csv")

        # set transforms
        if train_transforms is None:
            self.train_transforms = transforms.Compose([
                transforms.ToTensor()
            ])
        else:
            self.train_transforms = train_transforms

        if test_transforms is None:
            self.test_transforms = transforms.Compose([
                transforms.ToTensor()
            ])
        else:
            self.test_transforms = test_transforms

        # other parameters
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.val_fold = val_fold
        self.study_level = study_level
        self.study_level_validation = study_level_validation
        self.drop_last = drop_last
        self.cast_dataset = cast_dataset

    def prepare_data(self):
        pass

    def setup(self, stage=None):

        # Assign train datasets for use in dataloaders
        print("Preparing train set")
        self.mura_train = MURADataset(self.data_dir, self.train_df, transform=self.train_transforms,
                                          study_level=(self.study_level or self.study_level_validation))

        print("Preparing validation set")
        self.mura_val = MURADataset(self.data_dir, self.valid_df, transform=self.train_transforms,
                                          study_level=(self.study_level or self.study_level_validation))

        print("Preparing test set")
        # Assign test dataset for use in dataloader(s)
        self.mura_test = MURADataset(self.data_dir, self.test_csv, transform=self.test_transforms,
                                     study_level=(self.study_level or self.study_level_validation))

    def get_pos_weight(self):
        return (self.mura_train.__len__() - self.mura_train.__pos__()) / self.mura_train.__pos__()

    def train_dataloader(self):
        if self.study_level:
            mura_train_dl = DataLoader(self.mura_train,
                                       shuffle=True,
                                       batch_size=self.batch_size,
                                       num_workers=self.num_workers,
                                       collate_fn=partial(study_collate_fn, max_images=self.batch_size*3),
                                       drop_last=self.drop_last)
        elif self.study_level_validation:
            mura_train_dl = DataLoader(self.mura_train,
                                       shuffle=True,
                                       batch_size=self.batch_size,
                                       num_workers=self.num_workers,
                                       collate_fn=partial(images_from_study_collate_fn, max_images=self.batch_size*3),
                                       drop_last=self.drop_last)
        else:
            mura_train_dl = DataLoader(self.mura_train,
                                       shuffle=True,
                                       batch_size=self.batch_size,
                                       num_workers=self.num_workers,
                                       drop_last=self.drop_last)
        return mura_train_dl

    def val_dataloader(self):
        if self.study_level or self.study_level_validation:
            return DataLoader(self.mura_val, batch_size=self.batch_size, num_workers=self.num_workers,
                              collate_fn=partial(study_collate_fn, max_images=self.batch_size*3),
                              drop_last=self.drop_last)
        else:
            return DataLoader(self.mura_val, batch_size=self.batch_size, num_workers=self.num_workers,
                              drop_last=self.drop_last)

    def test_dataloader(self):
        if self.study_level or self.study_level_validation:
            return DataLoader(self.mura_test, batch_size=self.batch_size, num_workers=self.num_workers,
                              collate_fn=partial(study_collate_fn, max_images=self.batch_size*3))
        else:
            return DataLoader(self.mura_test, batch_size=self.batch_size, num_workers=self.num_workers)