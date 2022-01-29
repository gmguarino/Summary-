#!.venv/bin/python3
from sklearn.metrics import f1_score
import torch
import transformers
import pytorch_lightning as pl
import torch.nn as nn
import torchmetrics as tm

from datamodule import ArxivDataLoader
from datasets import load_dataset

import argparse
import logging

import sys
import os
import sh

import pdb

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
logger.addHandler(logging.StreamHandler(sys.stdout))

# To not have redundant logs
try:
    sh.rm('-r', '-f', 'logs')
except FileNotFoundError:
    logger.warning("No such file or directory 'logs', skipping removal.")
sh.mkdir('logs')


class EncoderClassifier(pl.LightningModule):

    def __init__(self, args) -> None:
        super().__init__()
        self.model = transformers.BertForSequenceClassification.from_pretrained(args.model, num_labels=args.num_labels)
        self.model_type = args.model
        self.batch_size = args.batch_size
        self.debug = args.debug
        self.dataset_percent = args.percent
        self.max_seq_len = args.max_len
        self.args=args
        self.loss=nn.CrossEntropyLoss(reduction="none")

    def process_data(self):

        tokenizer = transformers.BertTokenizer.from_pretrained(self.model_type)

        def _tokenize_dataset(dataset):
            dataset['input_ids'] = tokenizer.batch_encode_plus(
                dataset['text'], 
                max_length=self.max_seq_len, 
                pad_to_max_length=True)['input_ids']
            return dataset
        
        def _load_data(split):
            percent = f"{self.dataset_percent}%" if split=="train" else "100%"
            dataset_length = self.batch_size if self.debug else percent

            logger.debug(f"Dataset items to download: {dataset_length}")

            dataset = load_dataset("ccdv/arxiv-classification", split=f"{split}[:{dataset_length}")
            dataset = dataset.map(_tokenize_dataset, batched=True)
            dataset.set_format(type='torch', columns=['input_ids', 'attention_mask', 'label'])
            pdb.set_trace()
            
            return dataset
        
        self.train_ds, self.validation_ds = map(_load_data, ("train", "validation"))
    
    def forward(self, inputs):
        mask = (inputs != 0).float()
        logits, = self.model(inputs, mask).to_tuple()
        return logits

    @staticmethod
    def get_metrics(logits, labels, num_labels, average="micro", split="train"):
        accuracy = tm.functional.accuracy(logits, labels, num_classes=num_labels, average=average)
        precision = tm.functional.precision(logits, labels, num_classes=num_labels, average=average)
        recall = tm.functional.recall(logits, labels, num_classes=num_labels, average=average)
        f1_score = tm.functional.f1(logits, labels, num_classes=num_labels, average=average)
        return {f'{split}_accuracy':accuracy, f"{split}_precision":precision,f"{split}_recall":recall, f"{split}_f1_score":f1_score}

    def training_step(self, batch):
        pdb.set_trace()
        logits = self.forward(batch['input_ids'])
        loss = self.loss(logits, batch['label']).mean()
        return {
            'loss': loss, 
            'log': {
                'train_loss': loss,
                **EncoderClassifier.get_metrics(logits, batch["label"], self.args.num_labels, average="micro", split="train")
            }
        }

    def validation_step(self, batch):
        with torch.no_grad():
            logits = self.forward(batch['input_ids'], batch["attention_mask"])
        loss = self.loss(logits, batch['label']).mean()
        return {
            'loss': loss, 
            'log': {
                'train_loss': loss,
                **EncoderClassifier.get_metrics(logits, batch["label"], self.args.num_labels, average="micro", split="validation")
            }
        }

    def validation_epoch_end(self, outputs):
        return outputs

    

    def configure_optimizers(self):
        return torch.optim.SGD(
            self.parameters(),
            lr=self.args.lr,
            momentum=self.args.momentum,
        )

    def save_model(self):
        pass


def main(args):
    model = EncoderClassifier(args)
    arxiv_dm = ArxivDataLoader(
        args.max_len,
        args.percent,
        args.batch_size,
        args.debug,
        args.cache_dir,
        args.model
    )
    trainer = pl.Trainer(
        default_root_dir='logs',
        gpus=(torch.cuda.device_count() if torch.cuda.is_available() else 0),
        max_epochs=args.epochs,
        fast_dev_run=args.debug,
        logger=pl.loggers.TensorBoardLogger('logs/', name='clf', version=0)
    )
    trainer.fit(model, datamodule=arxiv_dm)


if __name__=="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--batch-size",
        type=int,
        default=32,
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
        "--max-len",
        type=int,
        default=256,
        metavar="N",
        help="Maximum sequence length of model inputs (default: 256)",
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
        "--momentum", type=float, default=0.5, metavar="M", help="SGD momentum (default: 0.5)"
    )
    parser.add_argument(
        "--percent", type=float, default=5, metavar="Percent", help="Percentage of training dataset to load (default: 5)"
    )

    parser.add_argument(
        "--model", type=str, default="bert-base-uncased", metavar="Model", help="default: bert-based-uncased"
    )
    parser.add_argument(
        "--cache-dir", type=str, default="../data", metavar="Cache", help=""
    )
    parser.add_argument(
        "--num-labels", type=int, default=11, metavar="N", help="default: 11"
    )

    args = parser.parse_args()

    main(args)