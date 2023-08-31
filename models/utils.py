import torch
from torch import nn

from torch.nn.utils.rnn import pack_sequence


def groupMax(features, seq_sizes):
    """
    max of features by group of images
    :param features: output of the feature extractor for all images in batch
    :param seq_sizes: sizes of original images sequences
    :return: max of features grouped by sequences
    """
    grouped_features = torch.zeros((len(seq_sizes), features.shape[1])).cuda()
    cumulative = 0
    for i in range(len(seq_sizes)):
        grouped_features[i] = torch.amax(features[cumulative:cumulative + seq_sizes[i]], axis=0)
        cumulative += seq_sizes[i]

    return grouped_features


def groupMean(features, seq_sizes):
    """
    max of features by group of images
    :param features: output of the feature extractor for all images in batch
    :param seq_sizes: sizes of original images sequences
    :return: max of features grouped by sequences
    """
    grouped_features = torch.zeros((len(seq_sizes), features.shape[1])).cuda()
    cumulative = 0
    for i in range(len(seq_sizes)):
        grouped_features[i] = torch.mean(features[cumulative:cumulative + seq_sizes[i]], axis=0)
        cumulative += seq_sizes[i]

    return grouped_features


def groupSeq(features, seq_sizes, shuffle=False):
    """
    group features of images by exams
    :param features: output of the feature extractor for all images in batch
    :param seq_sizes: sizes of original images sequences
    :param shuffle: shuffle the images inside a sequence (keep order between sequences), default False
    :return: features grouped by sequences
    """
    list_features = []
    cumulative = 0
    for i in range(len(seq_sizes)):
        if shuffle:
            perm = torch.randperm(seq_sizes[i])
            list_features.append(features[cumulative+perm])
        else:
            list_features.append(features[cumulative:cumulative + seq_sizes[i]])
        cumulative += seq_sizes[i]
    packed_features = pack_sequence(list_features, enforce_sorted=False)

    return packed_features


def groupFeatures(features, seq_sizes):
    """
    group features of images by exams
    :param features: output of the feature extractor for all images in batch
    :param seq_sizes: sizes of original images sequences
    :return: features grouped by sequences
    """
    n_studies = len(seq_sizes)
    n_features_per_image = features.shape[-1] * features.shape[-2]
    features_size = features.shape[-3]
    max_per_study = 11
    grouped_features = torch.zeros((n_studies, features_size, n_features_per_image*max_per_study)).cuda()

    cumulative = 0
    for i in range(len(seq_sizes)):
        cur_flat_size = n_features_per_image*seq_sizes[i]
        grouped_features[i, :, 0:cur_flat_size] \
            = features[cumulative:cumulative+seq_sizes[i]].swapaxes(0, 1).flatten(start_dim=1)
        cumulative += seq_sizes[i]

    return grouped_features


class UnList(nn.Module):
    def __init__(self, module):
        super(UnList, self).__init__()
        self.module = module
        self.fc = module.fc
        self.module.fc = nn.Identity()

    def forward(self, x):
        x = self.module(x)[0]
        return self.fc(x)


class LabelSmoothing(nn.Module):
    def __init__(self, loss_function, epsilon=0.1, out_features=1):
        super().__init__()
        self.epsilon = epsilon
        if out_features > 1:
            self.k = out_features - 1
        else:
            self.k = 1
        self.loss_function = loss_function

    def forward(self, input, target):
        smooth_target = target - self.epsilon*target + self.epsilon*(1-target)/self.k
        return self.loss_function(input, smooth_target)
