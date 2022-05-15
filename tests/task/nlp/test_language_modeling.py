import sys
from unittest.mock import MagicMock

import pytest
import pytorch_lightning as pl
from transformers import AutoTokenizer

from lightning_transformers.task.nlp.language_modeling import (
    LanguageModelingDataConfig,
    LanguageModelingDataModule,
    LanguageModelingTransformer,
)


@pytest.mark.skipif(sys.platform == "win32", reason="Currently Windows is not supported")
def test_smoke_train(hf_cache_path):
    tokenizer = AutoTokenizer.from_pretrained(pretrained_model_name_or_path="sshleifer/tiny-gpt2")
    model = LanguageModelingTransformer(pretrained_model_name_or_path="sshleifer/tiny-gpt2")
    dm = LanguageModelingDataModule(
        cfg=LanguageModelingDataConfig(
            batch_size=1,
            dataset_name="wikitext",
            dataset_config_name="wikitext-2-raw-v1",
            cache_dir=hf_cache_path,
        ),
        tokenizer=tokenizer,
    )
    trainer = pl.Trainer(fast_dev_run=True)

    trainer.fit(model, dm)


@pytest.mark.skipif(sys.platform == "win32", reason="Currently Windows is not supported")
def test_smoke_predict():
    model = LanguageModelingTransformer(
        pretrained_model_name_or_path="sshleifer/tiny-gpt2",
        tokenizer=AutoTokenizer.from_pretrained(pretrained_model_name_or_path="sshleifer/tiny-gpt2"),
    )
    y = model.hf_predict("The house:")
    assert len(y) == 1
    assert y[0]["generated_text"].startswith("The house:")


def test_model_has_correct_cfg():
    model = LanguageModelingTransformer(pretrained_model_name_or_path="sshleifer/tiny-gpt2")
    assert model.hparams.downstream_model_type == "transformers.AutoModelForCausalLM"


def test_datamodule_has_correct_cfg():
    tokenizer = MagicMock()
    dm = LanguageModelingDataModule(tokenizer)
    assert isinstance(dm.cfg, LanguageModelingDataConfig)
    assert dm.tokenizer is tokenizer
