import transformers
import pytorch_lightning as pl

from typing import Optional
from datasets import load_dataset
from torch.utils.data import DataLoader

import logging


logger = logging.getLogger(__name__)


class ArxivDataLoader(pl.LightningDataModule):

    def __init__(self, max_seq_len, dataset_percent, batch_size, debug, cache_dir, model_type):
        super().__init__()
        self.max_seq_len=max_seq_len
        self.dataset_percent=dataset_percent
        self.batch_size=batch_size
        self.debug=debug
        self.cache_dir=cache_dir
        self.model_type=model_type

    def setup(self, stage=None) -> None:
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

            dataset = load_dataset(
                "ccdv/arxiv-classification", 
                split=f"{split}[:{dataset_length}]",
                cache_dir=self.cache_dir)
            dataset = dataset.map(_tokenize_dataset, batched=True)
            dataset.set_format(type='torch', columns=['input_ids', 'label'])
            
            return dataset
        
        self.train_ds, self.validation_ds = map(_load_data, ("train", "validation"))

    def train_dataloader(self):
        return DataLoader(
            self.train_ds,
            self.batch_size,
            shuffle=True,
            drop_last=True
        )

    def val_dataloader(self):
        return DataLoader(
            self.validation_ds,
            self.batch_size,
            shuffle=False,
            drop_last=True
        )

    def teardown(self, stage=None):
        return super().teardown()