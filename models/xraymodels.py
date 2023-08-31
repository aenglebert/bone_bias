import pytorch_lightning as pl
from torchmetrics.classification.accuracy import Accuracy
from torchmetrics.classification.auroc import AUROC

from torch.optim.lr_scheduler import ReduceLROnPlateau, CosineAnnealingLR

import torch
from torch import nn
import torch.nn.functional as F

import math

from .utils import groupMax, groupMean, LabelSmoothing, groupFeatures


class Resnet50(pl.LightningModule):
    def __init__(
            self,
            out_features,
            encoder=None,
            imagenet_pretrain=False,
            lr: float = 6e-5,
            weight_decay: float = 1e-5,
            pos_weight: float = 1,
            max_epochs: int = 20,
            input_height: int = 320,
            batch_size: int = 16,
            scheduler="CosineAnnealingLR"
    ):
        super().__init__()
        if encoder is not None:
            self.encoder = encoder
        else:
            if imagenet_pretrain:
                weights='ResNet50_Weights.IMAGENET1K_V1'
            else:
                weights=None
            self.encoder = torch.hub.load('pytorch/vision', "resnet50", weights=weights)
            self.encoder.fc = nn.Identity()

        self.fc = nn.Linear(2048, out_features)

        if out_features == 1:
            self.out_fn = F.sigmoid
            task = "binary"

            self.criterion = torch.nn.BCEWithLogitsLoss(pos_weight=torch.tensor(pos_weight))
        else:
            self.out_fn = F.softmax
            self.criterion = torch.nn.NLLLoss()
            task = "multiclass"

        self.save_hyperparameters()

        self.train_acc = Accuracy(task)
        self.train_auroc = AUROC(task)
        self.val_acc = Accuracy(task)
        self.val_auroc = AUROC(task)
        self.test_group_max_metric = Accuracy(task)
        self.test_group_mean_metric = Accuracy(task)

        self.training_step_outputs = []
        self.validation_step_outputs = []
        self.test_step_outputs = []
        self.test_results = {}

        self._features = torch.empty(0)

    def set_hooks(self):
        self.feature_layer = dict(self.named_modules())['encoder.layer4']
        self.feature_layer.register_forward_hook(self.save_features_hook())

    def save_features_hook(self):
        def fn(_, __, output):
            self._features = output.data
        return fn

    def cam(self):
        with torch.no_grad():
            fc_weights = self.fc.weight.data.t().unsqueeze(0)
            b, c, h, w = self._features.size()
            features = self._features.view(b, c, h*w).transpose(1,2)
            cam = torch.bmm(features, fc_weights)
            cam = torch.relu(cam)
            # normalize to 0,1
            cam -= torch.min(cam)
            cam /= torch.max(cam)
            cam = cam.view(1, -1, h, w)
        return cam

    def forward(self, x):
        x = self.encoder(x)
        x = self.fc(x)
        return self.out_fn(x)

    def batch_forward(self, batch):
        x, y = batch
        out = self.fc(self.encoder(x))
        return out, y

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.hparams.lr, weight_decay=self.hparams.weight_decay)

        if self.hparams.scheduler == "ReduceLROnPlateau":
            scheduler = ReduceLROnPlateau(optimizer, mode='max', factor=0.1, patience=1, verbose=True)
        elif self.hparams.scheduler == "CosineAnnealingLR":
            scheduler = CosineAnnealingLR(optimizer, T_max=self.hparams.max_epochs, eta_min=self.hparams.lr/100, verbose=True)
        return {'optimizer': optimizer, 'lr_scheduler': scheduler, 'monitor': 'val_auroc'}

    def training_step(self, train_batch, batch_idx):
        out, y = self.batch_forward(train_batch)
        y = y.unsqueeze(1).float()
        loss = self.criterion(out, y)
        preds = self.out_fn(out)
        acc = self.train_acc(preds, y.int())
        auroc = self.train_auroc(preds, y.int())
        out = {'loss': loss, 'acc': acc, 'auroc': auroc}
        self.training_step_outputs.append(out)
        return out

    def log_histogram(self):
        # iterating through all parameters
        for name, params in self.named_parameters():
            self.logger.experiment.add_histogram(name, params, self.current_epoch)

    def on_train_epoch_end(self):
        #  called after every epoch
        # average loss and acc for the epoch
        outs = self.training_step_outputs
        avg_loss = torch.stack([x['loss'] for x in outs]).mean()
        avg_acc = self.train_acc.compute()
        auroc = self.train_auroc.compute()

        # logging
        if self.logger is not None:
            self.log("train_auroc", auroc)
            self.logger.experiment.add_scalar("Loss/Train", avg_loss, self.current_epoch)
            self.logger.experiment.add_scalar("Accuracy/Train", avg_acc, self.current_epoch)
            self.logger.experiment.add_scalar("AUROC/Train", auroc, self.current_epoch)
            self.log_histogram()

    def validation_step(self, val_batch, batch_idx):
        out, y = self.batch_forward(val_batch)
        y = y.unsqueeze(1).float()
        loss = self.criterion(out, y)
        preds = self.out_fn(out)
        acc = self.val_acc(preds, y.int())
        auroc = self.val_auroc(preds, y.int())
        outs = {'loss': loss, 'acc': acc, 'auroc': auroc}
        self.validation_step_outputs.append(outs)
        return outs

    def on_validation_epoch_end(self):
        #  called after every epoch
        # average loss and acc for the epoch
        outs = self.validation_step_outputs
        avg_loss = torch.stack([x['loss'] for x in outs]).mean()
        avg_acc = self.val_acc.compute()
        auroc = self.val_auroc.compute()

        # logging
        if self.logger is not None:
            self.log("val_auroc", auroc)
            self.logger.experiment.add_scalar("Loss/Validation", avg_loss, self.current_epoch)
            self.logger.experiment.add_scalar("Accuracy/Validation", avg_acc, self.current_epoch)
            self.logger.experiment.add_scalar("AUROC/Validation", auroc, self.current_epoch)
        return {'val_loss': avg_loss, 'val_acc': avg_acc, 'val_auroc': auroc}

    def test_step(self, test_batch, batch_idx):
        x, seq_sizes, y = test_batch
        features = self.fc(self.encoder(x))
        img_out = self.out_fn(features)
        mean_out = groupMean(img_out, seq_sizes)
        max_out = groupMax(img_out, seq_sizes)

        y = y.unsqueeze(1).float()
        mean_acc = self.test_group_mean_metric(mean_out, y.int())
        max_acc = self.test_group_max_metric(max_out, y.int())

        outs = {'mean_out': mean_out, 'max_out': max_out, 'test_max_acc': max_acc, 'test_mean_acc': mean_acc}
        self.test_step_outputs.append(outs)

        return outs

    def on_test_epoch_end(self):

        outs = self.test_step_outputs

        #  called after every epoch
        max_output = torch.cat([x['max_out'] for x in outs])
        mean_output = torch.cat([x['mean_out'] for x in outs])
        avg_max_acc = self.test_group_max_metric.compute()
        avg_mean_acc = self.test_group_mean_metric.compute()

        # logging
        if self.logger is not None:
            self.logger.experiment.add_scalar("Accuracy (group max)/Test", avg_max_acc, self.current_epoch)
            self.logger.experiment.add_scalar("Accuracy (group mean)/Test", avg_mean_acc, self.current_epoch)

        self.test_results = {'max_output': max_output.cpu().numpy(), 'mean_output': mean_output.cpu().numpy()}
