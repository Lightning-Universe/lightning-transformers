import os
import sys
from unittest.mock import MagicMock

import pytest
import pytorch_lightning as pl
import transformers
from pytorch_lightning.callbacks import ModelCheckpoint
from transformers import AutoTokenizer

from lightning_transformers.plugins.checkpoint import HFSaveCheckpoint
from lightning_transformers.task.nlp.text_classification import (
    TextClassificationDataModule,
    TextClassificationTransformer,
)


@pytest.mark.skipif(sys.platform == "win32", reason="Currently Windows is not supported")
def test_smoke_train(hf_cache_path):
    tokenizer = AutoTokenizer.from_pretrained(pretrained_model_name_or_path="prajjwal1/bert-tiny")
    dm = TextClassificationDataModule(
        batch_size=1,
        dataset_name="glue",
        dataset_config_name="sst2",
        max_length=512,
        limit_test_samples=64,
        limit_val_samples=64,
        limit_train_samples=64,
        cache_dir=hf_cache_path,
        tokenizer=tokenizer,
    )
    model = TextClassificationTransformer(pretrained_model_name_or_path="prajjwal1/bert-tiny")
    trainer = pl.Trainer(fast_dev_run=True)
    trainer.fit(model, dm)


def test_smoke_predict_with_trainer(hf_cache_path):
    tokenizer = AutoTokenizer.from_pretrained(pretrained_model_name_or_path="prajjwal1/bert-tiny")
    dm = TextClassificationDataModule(
        batch_size=1,
        dataset_name="glue",
        dataset_config_name="sst2",
        max_length=512,
        limit_test_samples=64,
        limit_val_samples=64,
        limit_train_samples=64,
        cache_dir=hf_cache_path,
        predict_subset_name="test",  # Use the "test" split of the dataset as our prediction subset
        tokenizer=tokenizer,
    )
    model = TextClassificationTransformer(pretrained_model_name_or_path="prajjwal1/bert-tiny")
    trainer = pl.Trainer(fast_dev_run=True)
    y = trainer.predict(model, dm)
    assert len(y) == 1
    assert int(y[0]) in (0, 1)


@pytest.mark.skipif(sys.platform == "win32", reason="Currently Windows is not supported")
def test_smoke_predict():
    model = TextClassificationTransformer(
        pretrained_model_name_or_path="prajjwal1/bert-tiny",
        tokenizer=AutoTokenizer.from_pretrained(pretrained_model_name_or_path="prajjwal1/bert-tiny"),
    )
    y = model.hf_predict("Lightning rocks!")
    assert len(y) == 1
    assert isinstance(y[0]["score"], float)


def test_model_has_correct_cfg():
    model = TextClassificationTransformer(pretrained_model_name_or_path="prajjwal1/bert-tiny")
    assert isinstance(model.model, transformers.BertForSequenceClassification)


def test_datamodule_has_tokenizer():
    tokenizer = MagicMock()
    dm = TextClassificationDataModule(tokenizer)
    assert dm.tokenizer is tokenizer


@pytest.mark.skipif(sys.platform == "win32", reason="Currently Windows is not supported")
def test_huggingface_checkpoint_train(hf_cache_path, tmpdir):
    tokenizer = AutoTokenizer.from_pretrained(pretrained_model_name_or_path="prajjwal1/bert-tiny")
    dm = TextClassificationDataModule(
        batch_size=1,
        dataset_name="glue",
        dataset_config_name="sst2",
        max_length=512,
        limit_test_samples=64,
        limit_val_samples=64,
        limit_train_samples=64,
        cache_dir=hf_cache_path,
        tokenizer=tokenizer,
    )
    ckpt_path = os.path.join(tmpdir, "checkpoints")
    model = TextClassificationTransformer(pretrained_model_name_or_path="prajjwal1/bert-tiny")
    trainer = pl.Trainer(
        default_root_dir=tmpdir,
        max_epochs=1,
        limit_train_batches=1,
        limit_val_batches=1,
        plugins=HFSaveCheckpoint(model=model),
        callbacks=ModelCheckpoint(save_last=True, dirpath=ckpt_path),
    )
    trainer.fit(model, dm)
    assert os.path.exists(os.path.join(ckpt_path, "last_huggingface"))
