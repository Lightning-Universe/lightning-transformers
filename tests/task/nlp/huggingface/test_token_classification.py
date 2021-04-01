import sys
from unittest.mock import MagicMock

import pytest

from lightning_transformers.core.nlp.huggingface import HFBackboneConfig
from lightning_transformers.task.nlp.token_classification import (
    TokenClassificationDataModule,
    TokenClassificationTransformer,
)
from lightning_transformers.task.nlp.token_classification.config import TokenClassificationDataConfig


@pytest.mark.skipif(sys.platform == "win32", reason="Currently Windows is not supported")
def test_smoke_train_e2e(script_runner):
    script_runner.hf_train(task='token_classification', dataset='conll', model='prajjwal1/bert-tiny')


def test_smoke_predict_e2e(script_runner):
    y = script_runner.hf_predict(
        ['+x="Have a good day!"', '+model_data_kwargs={labels: 2}'],
        task='token_classification',
        model='prajjwal1/bert-tiny',
    )
    assert len(y) == 5
    assert [a['word'] for a in y] == ['have', 'a', 'good', 'day', '!']


def test_model_has_correct_cfg():
    model = TokenClassificationTransformer(
        HFBackboneConfig(pretrained_model_name_or_path='bert-base-cased'),
        labels=2,
    )
    assert model.hparams.downstream_model_type == 'transformers.AutoModelForTokenClassification'
    assert model.num_labels == 2


def test_datamodule_has_correct_cfg():
    tokenizer = MagicMock()
    dm = TokenClassificationDataModule(tokenizer)
    assert type(dm.cfg) is TokenClassificationDataConfig
    assert dm.tokenizer is tokenizer
