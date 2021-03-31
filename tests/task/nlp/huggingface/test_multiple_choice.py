import sys
from unittest.mock import MagicMock

import pytest

from lightning_transformers.core.nlp.huggingface import HFBackboneConfig, HFTransformerDataConfig
from lightning_transformers.task.nlp.multiple_choice import MultipleChoiceDataModule, MultipleChoiceTransformer


@pytest.mark.skipif(sys.platform == "win32", reason="Currently Windows is not supported")
def test_smoke_train_e2e(script_runner):
    script_runner.hf_train(task='multiple_choice', dataset='swag', model='prajjwal1/bert-tiny')


def test_model_can_be_created():
    MultipleChoiceTransformer(
        'transformers.AutoModelForMultipleChoice',
        HFBackboneConfig(pretrained_model_name_or_path='bert-base-cased'),
    )


def test_datamodule_has_correct_cfg():
    tokenizer = MagicMock()
    dm = MultipleChoiceDataModule(tokenizer)
    assert type(dm.cfg) is HFTransformerDataConfig
    assert dm.tokenizer is tokenizer
