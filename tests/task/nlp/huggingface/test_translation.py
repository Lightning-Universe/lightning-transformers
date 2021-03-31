import sys
from unittest.mock import MagicMock

import pytest
import pytorch_lightning as pl
import torch
from transformers import AutoTokenizer

from examples.custom.dataset.translation.custom_dataset import MyTranslationDataModule
from lightning_transformers.core.nlp.huggingface import HFBackboneConfig
from lightning_transformers.task.nlp.translation import TranslationTransformer, WMT16TranslationDataModule
from lightning_transformers.task.nlp.translation.config import TranslationConfig, TranslationDataConfig
from lightning_transformers.task.nlp.translation.data import TranslationDataModule


@pytest.mark.skipif(sys.platform == "win32", reason="Currently Windows is not supported")
def test_smoke_train_e2e(script_runner):
    script_runner.hf_train(task='translation', dataset='wmt16', model='patrickvonplaten/t5-tiny-random')


def test_smoke_predict_e2e(script_runner):
    y = script_runner.hf_predict(['+x="¡Hola Sean!"'], task='translation', model='patrickvonplaten/t5-tiny-random')
    assert len(y) == 1
    assert isinstance(y[0]['translation_text'], str)


def test_model_has_correct_cfg():
    model = TranslationTransformer(HFBackboneConfig(pretrained_model_name_or_path='patrickvonplaten/t5-tiny-random'))
    assert model.hparams.downstream_model_type == 'transformers.AutoModelForSeq2SeqLM'
    assert type(model.cfg) is TranslationConfig


def test_datamodule_has_correct_cfg():
    tokenizer = MagicMock()
    dm = TranslationDataModule(tokenizer)
    assert type(dm.cfg) is TranslationDataConfig
    assert dm.tokenizer is tokenizer


@pytest.mark.parametrize("cls", [WMT16TranslationDataModule, MyTranslationDataModule])
def test_non_hydra_model(cls, hf_cache_path):

    class MyTranslationTransformer(TranslationTransformer):

        def configure_optimizers(self):
            return torch.optim.AdamW(self.parameters(), lr=1e-5)

    tokenizer = AutoTokenizer.from_pretrained(pretrained_model_name_or_path='patrickvonplaten/t5-tiny-random')

    model = MyTranslationTransformer(
        backbone=HFBackboneConfig(pretrained_model_name_or_path='patrickvonplaten/t5-tiny-random')
    )

    dm = WMT16TranslationDataModule(
        cfg=TranslationDataConfig(
            batch_size=1,
            dataset_name='wmt16',
            dataset_config_name='ro-en',
            source_language='en',
            target_language='ro',
            cache_dir=hf_cache_path,
            limit_train_samples=16,
            limit_val_samples=16,
            limit_test_samples=16,
            max_source_length=32,
            max_target_length=32
        ),
        tokenizer=tokenizer
    )

    trainer = pl.Trainer(fast_dev_run=True)

    trainer.fit(model, dm)
