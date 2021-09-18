import sys
from unittest.mock import MagicMock

import pytest
import pytorch_lightning as pl
import torch
from transformers import AutoTokenizer

from lightning_transformers.core.nlp import HFBackboneConfig
from lightning_transformers.task.nlp.masked_language_modeling import (
    MaskedLanguageModelingDataModule,
    MaskedLanguageModelingTransformer,
)
from lightning_transformers.task.nlp.masked_language_modeling.config import MaskedLanguageModelingDataConfig


def test_smoke_train(hf_cache_path):
    class TestModel(MaskedLanguageModelingTransformer):
        def configure_optimizers(self):
            return torch.optim.AdamW(self.parameters(), lr=1e-5)

    tokenizer = AutoTokenizer.from_pretrained(pretrained_model_name_or_path="prajjwal1/bert-tiny")
    model = TestModel(backbone=HFBackboneConfig(pretrained_model_name_or_path="prajjwal1/bert-tiny"))
    dm = MaskedLanguageModelingDataModule(
        cfg=MaskedLanguageModelingDataConfig(
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
def test_smoke_train_e2e(script_runner):
    script_runner.hf_train(task="masked_language_modeling", dataset="wikitext", model="prajjwal1/bert-tiny")


def test_smoke_predict_e2e(script_runner):
    y = script_runner.hf_predict(
        ['+x="The cat sat on [MASK] mat."'], task="masked_language_modeling", model="prajjwal1/bert-tiny"
    )
    assert len(y) == 5
    assert y[0]["token_str"] == "the"


def test_model_has_correct_cfg():
    model = MaskedLanguageModelingTransformer(HFBackboneConfig(pretrained_model_name_or_path="prajjwal1/bert-tiny"))
    assert model.hparams.downstream_model_type == "transformers.AutoModelForMaskedLM"


def test_datamodule_has_correct_cfg():
    tokenizer = MagicMock()
    dm = MaskedLanguageModelingDataModule(tokenizer)
    assert type(dm.cfg) is MaskedLanguageModelingDataConfig
    assert dm.tokenizer is tokenizer
