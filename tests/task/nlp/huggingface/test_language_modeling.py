import sys
from unittest.mock import MagicMock

import pytest
import pytorch_lightning as pl
from transformers import AutoTokenizer

from lightning_transformers.core.nlp.huggingface import HFBackboneConfig
from lightning_transformers.task.nlp.language_modeling import LanguageModelingDataModule, LanguageModelingTransformer
from lightning_transformers.task.nlp.language_modeling.config import LanguageModelingDataConfig


@pytest.mark.skipif(sys.platform == "win32", reason="Currently Windows is not supported")
def test_smoke_train_e2e(script_runner):
    script_runner.hf_train(task='language_modeling', dataset='wikitext', model='prajjwal1/bert-tiny')


def test_smoke_predict_e2e(script_runner):
    y = script_runner.hf_predict(['+x="The house:"'], task='language_modeling', model='sshleifer/tiny-gpt2')
    assert len(y) == 1
    assert y[0]['generated_text'].startswith("The house:")


def test_model_has_correct_cfg():
    model = LanguageModelingTransformer(HFBackboneConfig(pretrained_model_name_or_path='sshleifer/tiny-gpt2'))
    assert model.hparams.downstream_model_type == 'transformers.AutoModelForCausalLM'


def test_datamodule_has_correct_cfg():
    tokenizer = MagicMock()
    dm = LanguageModelingDataModule(tokenizer)
    assert type(dm.cfg) is LanguageModelingDataConfig
    assert dm.tokenizer is tokenizer


def test_smoke_train(hf_cache_path):
    tokenizer = AutoTokenizer.from_pretrained(pretrained_model_name_or_path='sshleifer/tiny-gpt2')
    model = LanguageModelingTransformer(backbone=HFBackboneConfig(pretrained_model_name_or_path='sshleifer/tiny-gpt2'))
    dm = LanguageModelingDataModule(
        cfg=LanguageModelingDataConfig(
            batch_size=1,
            dataset_name='wikitext',
            dataset_config_name='wikitext-2-raw-v1',
            cache_dir=hf_cache_path,
        ),
        tokenizer=tokenizer
    )
    trainer = pl.Trainer(fast_dev_run=True)

    trainer.fit(model, dm)
