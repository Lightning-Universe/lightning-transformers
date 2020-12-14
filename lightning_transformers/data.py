from typing import Optional

import pytorch_lightning as pl
from datasets import load_dataset
from torch.utils.data import DataLoader
from transformers import AutoTokenizer


def convert_to_features(example_batch, indices, tokenizer, text_fields, padding, truncation, max_length):
    # Either encode single sentence or sentence pairs
    if len(text_fields) > 1:
        texts_or_text_pairs = list(zip(example_batch[text_fields[0]], example_batch[text_fields[1]]))
    else:
        texts_or_text_pairs = example_batch[text_fields[0]]

    # Tokenize the text/text pairs
    features = tokenizer.batch_encode_plus(
        texts_or_text_pairs, padding=padding, truncation=truncation, max_length=max_length
    )

    # idx is unique ID we can use to link predictions to original data
    features['idx'] = indices

    return features


def preprocess(ds, tokenizer, text_fields, padding='max_length', truncation='only_first', max_length=128):
    ds = ds.map(
        convert_to_features,
        batched=True,
        with_indices=True,
        fn_kwargs={
            'tokenizer': tokenizer,
            'text_fields': text_fields,
            'padding': padding,
            'truncation': truncation,
            'max_length': max_length,
        },
    )
    ds.rename_column_('label', "labels")
    return ds


def transform_labels(example, idx, label2id: dict):
    str_label = example['labels']
    example['labels'] = label2id[str_label]
    example['idx'] = idx
    return example


class LitTransformerDataModule(pl.LightningDataModule):
    def __init__(
            self,
            dataset_name: str,
            subset_name: str,
            tokenizer: str = AutoTokenizer,
            padding: str = 'max_length',
            truncation: str = 'only_first',
            max_length: int = 128,
            batch_size: int = 16,
            num_workers: int = 8,
            train_val_split: Optional[int] = None,
            use_fast: bool = True):
        super().__init__()
        self.dataset_name = dataset_name
        self.subset_name = subset_name
        self.tokenizer = tokenizer
        self.padding = padding
        self.truncation = truncation
        self.max_length = max_length
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.train_val_split = train_val_split
        self.use_fast = use_fast

    def setup(self, stage: Optional[str] = None):

        self.ds = load_dataset(self.dataset_name, self.subset_name)

        if self.train_val_split is not None:
            split = self.ds['train'].train_test_split(self.train_val_split)
            self.ds['train'] = split['train']
            self.ds['validation'] = split['test']

        self.ds = preprocess(self.ds, self.tokenizer, self.text_fields, self.padding, self.truncation, self.max_length)

        if self.do_transform_labels:
            self.ds = self.ds.map(transform_labels, with_indices=True, fn_kwargs={'label2id': self.label2id})

        cols_to_keep = [
            x
            for x in ['input_ids', 'attention_mask', 'token_type_ids', 'labels', 'idx']
            if x in self.ds['train'].features
        ]
        self.ds.set_format("torch", columns=cols_to_keep)

    def train_dataloader(self):
        return DataLoader(self.ds['train'], batch_size=self.batch_size, num_workers=self.num_workers)

    def val_dataloader(self):
        return DataLoader(self.ds['validation'], batch_size=self.batch_size, num_workers=self.num_workers)

    def test_dataloader(self):
        return DataLoader(self.ds['test'], batch_size=self.batch_size, num_workers=self.num_workers)


# TODO expose all these variables and remove ones that are not required. A lot of these can be infered from the underlying dataset obj
class EmotionDataModule(LitTransformerDataModule):
    dataset_name = 'emotion'
    subset_name = None
    text_fields = ['text']
    label2id = {"sadness": 0, "joy": 1, "love": 2, "anger": 3, "fear": 4, "surprise": 5}
    do_transform_labels = False
    train_val_split = None


class AGNewsDataModule(LitTransformerDataModule):
    dataset_name = 'ag_news'
    subset_name = None
    text_fields = ['text']
    label2id = {"World": 0, "Sports": 1, "Business": 2, "Sci/Tech": 3}
    do_transform_labels = False
    train_val_split = 20000


class MrpcDataModule(LitTransformerDataModule):
    dataset_name = 'glue'
    subset_name = 'mrpc'
    text_fields = ['sentence1', 'sentence2']
    label2id = {"not_equivalent": 0, "equivalent": 1}
    do_transform_labels = False
    train_val_split = None
