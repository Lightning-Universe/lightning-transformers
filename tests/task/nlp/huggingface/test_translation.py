import sys
from unittest.mock import MagicMock

import pytest

from lightning_transformers.core.nlp.huggingface import HFBackboneConfig
from lightning_transformers.task.nlp.translation import TranslationTransformer
from lightning_transformers.task.nlp.translation.config import TranslationDataConfig, TranslationTransformerConfig
from lightning_transformers.task.nlp.translation.data import TranslationDataModule


@pytest.mark.skipif(sys.platform == "win32", reason="Currently Windows is not supported")
def test_smoke_train_e2e(script_runner):
    script_runner.hf_train(task='translation', dataset='wmt16', model='patrickvonplaten/t5-tiny-random')


def test_smoke_predict_e2e(script_runner):
    y = script_runner.hf_predict(['+x="¡Hola Sean!"'], task='translation', model='patrickvonplaten/t5-tiny-random')
    assert len(y) == 1
    assert isinstance(y[0]['translation_text'], str)


def test_model_has_correct_cfg():
    model = TranslationTransformer(
        HFBackboneConfig(
            downstream_model_type='transformers.AutoModelForSeq2SeqLM', pretrained_model_name_or_path='t5-base'
        ),
    )
    assert type(model.cfg) is TranslationTransformerConfig


def test_datamodule_has_correct_cfg():
    tokenizer = MagicMock()
    dm = TranslationDataModule(tokenizer)
    assert type(dm.cfg) is TranslationDataConfig
    assert dm.tokenizer is tokenizer
