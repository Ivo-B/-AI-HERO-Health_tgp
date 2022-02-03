from typing import Any, List

import torch
from torch import nn
from pytorch_lightning import LightningModule
from torchmetrics import MaxMetric
#from torchmetrics.classification.accuracy import Accuracy
from torchmetrics import Precision, Recall, Accuracy, AveragePrecision


from cvnets.models.classification import build_classification_model
import argparse
import yaml
import collections
from types import SimpleNamespace


def flatten_yaml_as_dict(d, parent_key='', sep='.'):
    items = []
    for k, v in d.items():
        new_key = parent_key + sep + k if parent_key else k
        if isinstance(v, collections.MutableMapping):
            items.extend(flatten_yaml_as_dict(v, new_key, sep=sep).items())
        else:
            items.append((new_key, v))
    return dict(items)


class NestedNamespace(SimpleNamespace):
    def __init__(self, dictionary, **kwargs):
        super().__init__(**kwargs)
        for key, value in dictionary.items():
            if isinstance(value, dict):
                self.__setattr__(key, NestedNamespace(value))
            else:
                self.__setattr__(key, value)


def load_config(config_file_name="/hkfs/work/workspace/scratch/im9193-H5/AI-HERO-Health_tgp/pre_trained/model_conf.yaml"):
    with open(config_file_name, 'r') as yaml_file:
        try:
            cfg = yaml.load(yaml_file, Loader=yaml.FullLoader)
            flat_cfg = flatten_yaml_as_dict(cfg)
            obj_cfg = NestedNamespace(flat_cfg)

        except yaml.YAMLError as exc:
            if is_master_node:
                logger.warning(
                    "Error while loading config file: {}".format(config_file_name)
                )
                logger.warning("Error message: {}".format(str(exc)))
    return obj_cfg


class MVITLitModule(LightningModule):
    """
    Example of LightningModule for MNIST classification.

    A LightningModule organizes your PyTorch code into 5 sections:
        - Computations (init).
        - Train loop (training_step)
        - Validation loop (validation_step)
        - Test loop (test_step)
        - Optimizers (configure_optimizers)

    Read the docs:
        https://pytorch-lightning.readthedocs.io/en/latest/common/lightning_module.html
    """

    def __init__(
        self,
        output_size: int = 1,
        pos_weight: int = 10,
        lr: float = 0.001,
        weight_decay: float = 0.0005,
        lr_scheduler_min_lr: float = 0.0005,
        lr_scheduler_factor: float = 0.0005,
        lr_scheduler_patience: float = 0.0005,
    ):
        super().__init__()

        # this line allows to access init params with 'self.hparams' attribute
        # it also ensures init params will be stored in ckpt
        self.save_hyperparameters(logger=False)

        self.model = build_classification_model(load_config())
        # self.model.classifier[-1] = nn.Linear(640, self.hparams.output_size)

        if self.hparams.freeze_layers:
            for param in self.model.parameters():
                param.requires_grad = False

            for param in self.model.layer_5.parameters():
                param.requires_grad = True

            for param in self.model.conv_1x1_exp.parameters():
                param.requires_grad = True

        self.model.classifier[-1] = nn.Linear(640, 32)
        self.model.classifier.add_module("Act SiLU", nn.SiLU())
        self.model.classifier.add_module("Dropout", nn.Dropout(p=0.5))
        self.model.classifier.add_module(
            "Output", nn.Linear(32, self.hparams.output_size)
        )

        # loss function
        self.criterion = torch.nn.BCEWithLogitsLoss(pos_weight=torch.FloatTensor([self.hparams.pos_weight]))
        # self.criterion = torch.nn.BCEWithLogitsLoss()

        # use separate metric instance for train, val and test step
        # to ensure a proper reduction over the epoch
        self.train_acc = Accuracy()
        self.train_precision = AveragePrecision()
        self.train_recall = Recall()

        self.val_acc = Accuracy()

        # for logging best so far validation accuracy
        self.val_acc_best = MaxMetric()

    def forward(self, x: torch.Tensor):
        return self.model(x)

    def step(self, batch: Any):
        x, y = batch
        logits = self.forward(x).squeeze()
        loss = self.criterion(logits, y)
        preds = (logits > 0).long()
        return loss, preds, y

    def training_step(self, batch: Any, batch_idx: int):
        loss, preds, targets = self.step(batch)

        # accumulate and return metrics for logging
        acc = self.train_acc(preds, targets.long())
        #print(acc)
        self.log("train/acc", acc, on_step=False, on_epoch=True, prog_bar=True)

        pre = self.train_precision(preds, targets.long())
        #print(pre)
        self.log("train/pre", pre, on_step=False, on_epoch=True, prog_bar=True)

        rec = self.train_recall(preds, targets.long())
        # print(rec)
        self.log("train/rec", rec, on_step=False, on_epoch=True, prog_bar=True)

        # we can return here dict with any tensors
        # and then read it in some callback or in `training_epoch_end()`` below
        # remember to always return loss from `training_step()` or else backpropagation will fail!
        return {"loss": loss, "preds": preds, "targets": targets}

    def training_epoch_end(self, outputs: List[Any]):
        # `outputs` is a list of dicts returned from `training_step()`
        pass

    def validation_step(self, batch: Any, batch_idx: int):
        loss, preds, targets = self.step(batch)

        # log val metrics
        self.log("val/loss", loss, on_step=False, on_epoch=True, prog_bar=False)

        acc = self.val_acc(preds, targets.long())
        self.log("val/acc", acc, on_step=False, on_epoch=True, prog_bar=True)

        sch = self.lr_schedulers()
        sch.step(loss)

        return {"loss": loss, "preds": preds, "targets": targets}

    def validation_epoch_end(self, outputs: List[Any]):
        acc = self.val_acc.compute()  # get val accuracy from current epoch
        self.log("val/acc", acc, on_step=False, on_epoch=True, prog_bar=True)
        self.val_acc.reset()

        self.val_acc_best.update(acc)
        self.log(
            "val/acc_best", self.val_acc_best.compute(), on_epoch=True, prog_bar=True
        )

    def test_step(self, batch: Any, batch_idx: int):
        loss, preds, targets = self.step(batch)

        # log test metrics
        acc = self.test_acc(preds, targets)
        self.log("test/loss", loss, on_step=False, on_epoch=True)
        self.log("test/acc", acc, on_step=False, on_epoch=True)

        return {"loss": loss, "preds": preds, "targets": targets}

    def test_epoch_end(self, outputs: List[Any]):
        pass

    def on_epoch_end(self):
        # reset metrics at the end of every epoch
        self.train_acc.reset()
        self.train_precision.reset()

        # self.test_acc.reset()
        self.val_acc.reset()

    def predict_step(self, batch, batch_idx, dataloader_idx=0):
        # this calls forward
        x, y = batch
        predicted = (self(x) > 0).long()
        return [[i for i in y], predicted.cpu().numpy()]


    def configure_optimizers(self):
        """Choose what optimizers and learning-rate schedulers to use in your optimization.
        Normally you'd need one. But in the case of GANs or similar you might have multiple.

        See examples here:
            https://pytorch-lightning.readthedocs.io/en/latest/common/lightning_module.html#configure-optimizers
        """
        optimizer = torch.optim.Adam(
            params=self.parameters(),
            lr=self.hparams.lr,
            weight_decay=self.hparams.weight_decay,
        )
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            "min",
            factor=self.hparams.lr_scheduler_factor,
            patience=self.hparams.lr_scheduler_patience,
            min_lr=self.hparams.lr_scheduler_min_lr,
        )
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "monitor": "val/loss",
            },
        }
