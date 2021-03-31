import sys
from unittest.mock import MagicMock

import pytest
from pytorch_lightning import seed_everything

from lightning_transformers.core.nlp.huggingface import HFBackboneConfig
from lightning_transformers.task.nlp.question_answering import (
    QuestionAnsweringTransformer,
    QuestionAnsweringTransformerDataModule,
)
from lightning_transformers.task.nlp.question_answering.config import QuestionAnsweringTransformerDataConfig


@pytest.mark.skipif(sys.platform == "win32", reason="Currently Windows is not supported")
def test_smoke_train_e2e(script_runner):
    script_runner.hf_train(task='question_answering', dataset='squad', model='prajjwal1/bert-tiny')


def test_smoke_predict_e2e(script_runner):
    seed_everything(0)
    y = script_runner.hf_predict(
        ['+x={context: "Lightning is great", question: "What is great?"}'],
        task='question_answering',
        model='prajjwal1/bert-tiny',
    )
    assert isinstance(y, dict)
    assert 'Lightning' in y['answer']


def test_model_can_be_created():
    QuestionAnsweringTransformer(
        'transformers.AutoModelForQuestionAnswering',
        HFBackboneConfig(pretrained_model_name_or_path='bert-base-cased'),
    )


def test_datamodule_has_correct_cfg():
    tokenizer = MagicMock()
    dm = QuestionAnsweringTransformerDataModule(tokenizer)
    assert type(dm.cfg) is QuestionAnsweringTransformerDataConfig
    assert dm.tokenizer is tokenizer
