import sys
from unittest.mock import MagicMock

import pytest
import pytorch_lightning as pl
import transformers
from transformers import AutoTokenizer

from lightning_transformers.task.nlp.masked_language_modeling import (
    MaskedLanguageModelingDataModule,
    MaskedLanguageModelingTransformer,
)


@pytest.mark.skipif(sys.platform == "win32", reason="Currently Windows is not supported")
def test_smoke_train(hf_cache_path):
    tokenizer = AutoTokenizer.from_pretrained(pretrained_model_name_or_path="prajjwal1/bert-tiny")
    model = MaskedLanguageModelingTransformer(pretrained_model_name_or_path="prajjwal1/bert-tiny")
    dm = MaskedLanguageModelingDataModule(
        batch_size=1,
        dataset_name="wikitext",
        dataset_config_name="wikitext-2-raw-v1",
        cache_dir=hf_cache_path,
        tokenizer=tokenizer,
    )
    trainer = pl.Trainer(fast_dev_run=True)

    trainer.fit(model, dm)


@pytest.mark.skipif(sys.platform == "win32", reason="Currently Windows is not supported")
def test_smoke_predict():
    model = MaskedLanguageModelingTransformer(
        pretrained_model_name_or_path="prajjwal1/bert-tiny",
        tokenizer=AutoTokenizer.from_pretrained(pretrained_model_name_or_path="prajjwal1/bert-tiny"),
    )
    y = model.hf_predict("The cat sat on [MASK] mat.")
    assert len(y) == 5
    assert y[0]["token_str"] == "the"


def test_model_has_correct_cfg():
    model = MaskedLanguageModelingTransformer(pretrained_model_name_or_path="prajjwal1/bert-tiny")
    assert isinstance(model.model, transformers.BertForMaskedLM)


def test_datamodule_has_tokenizer():
    tokenizer = MagicMock()
    dm = MaskedLanguageModelingDataModule(tokenizer)
    assert dm.tokenizer is tokenizer
