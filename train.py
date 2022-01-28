import torch
import datasets
import transformers
import pytorch_lightning as pl
import torch.nn as nn

import argparse
import logging

import sys
import os
import sh

import pdb

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
logger.addHandler(logging.StreamHandler(sys.stdout))


class EncoderClassifier(pl.LightningModule):

    def __init__(self, args) -> None:
        super().__init__()
        self.model = transformers.BertForSequenceClassification.from_pretrained(args.model)
        self.loss=None
        self.train_data=None
        self.val_data=None

    def process_data(self):
        pass
    
    def forward(self, inputs):
        pass

    def training_step(self):
        pass

    def training_epoch_end(self):
        pass

    def validation_step(self):
        pass

    def validation_epoch_end(self):
        pass

    def train_dataloader(self):
        pass

    def val_dataloader(self):
        pass

    def configure_optimizers(self):
        pass

    def save_model(self):
        pass


def main(args):
    model = EncoderClassifier()
    trainer = pl.Trainer(
        default_root_dir='logs',
        gpus=(1 if th.cuda.is_available() else 0),
        max_epochs=args.epochs,
        fast_dev_run=args.debug,
        logger=pl.loggers.TensorBoardLogger('logs/', name='clf', version=0),
    )
    trainer.fit(model)


if __name__=="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--batch-size",
        type=int,
        default=64,
        metavar="N",
        help="input batch size for training (default: 64)",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=10,
        metavar="N",
        help="number of epochs to train (default: 10)",
    )
    parser.add_argument(
        "--debug", 
        type=bool, 
        default=False, 
        metavar="Model", 
        help="Trains and validates on a single epoch if True (default: False)"
    )
    parser.add_argument(
        "--lr", type=float, default=0.00009, metavar="LR", help="learning rate (default: 0.00009)"
    )

    parser.add_argument(
        "--model", type=str, default="bert-based-uncased", metavar="Model", help="default: bert-based-uncased"
    )

    args = parser.parse_args()

    main(args)