import sys
from unittest.mock import MagicMock

import pytest
import pytorch_lightning as pl
import transformers
from transformers import AutoTokenizer

from lightning_transformers.task.nlp.language_modeling import (
    LanguageModelingDataConfig,
    LanguageModelingDataModule,
    LanguageModelingTransformer,
)


@pytest.mark.skipif(sys.platform == "win32", reason="Currently Windows is not supported")
@pytest.mark.parametrize("streaming", [True, False])
def test_smoke_train(hf_cache_path, streaming):
    tokenizer = AutoTokenizer.from_pretrained(pretrained_model_name_or_path="prajjwal1/bert-tiny")
    model = LanguageModelingTransformer(pretrained_model_name_or_path="prajjwal1/bert-tiny")
    dm = LanguageModelingDataModule(
        cfg=LanguageModelingDataConfig(
            batch_size=1,
            dataset_name="wikitext",
            dataset_config_name="wikitext-2-raw-v1",
            cache_dir=hf_cache_path,
            streaming=streaming,
        ),
        tokenizer=tokenizer,
    )
    trainer = pl.Trainer(max_steps=10, max_epochs=1, limit_val_batches=1, limit_test_batches=1, limit_train_batches=1)

    trainer.fit(model, dm)


@pytest.mark.skipif(sys.platform == "win32", reason="Currently Windows is not supported")
def test_smoke_predict():
    model = LanguageModelingTransformer(
        pretrained_model_name_or_path="prajjwal1/bert-tiny",
        tokenizer=AutoTokenizer.from_pretrained(pretrained_model_name_or_path="prajjwal1/bert-tiny"),
    )
    y = model.hf_predict("The house:")
    assert len(y) == 1
    assert y[0]["generated_text"].startswith("The house:")


def test_model_has_correct_cfg():
    model = LanguageModelingTransformer(pretrained_model_name_or_path="prajjwal1/bert-tiny")
    assert isinstance(model.model, transformers.BertLMHeadModel)


def test_datamodule_has_correct_cfg():
    tokenizer = MagicMock()
    dm = LanguageModelingDataModule(tokenizer)
    assert isinstance(dm.cfg, LanguageModelingDataConfig)
    assert dm.tokenizer is tokenizer
