import sys
from unittest.mock import MagicMock

import pytest
from pytorch_lightning import seed_everything

from lightning_transformers.core.nlp import HFBackboneConfig
from lightning_transformers.task.nlp.question_answering import QuestionAnsweringDataModule, QuestionAnsweringTransformer
from lightning_transformers.task.nlp.question_answering.config import QuestionAnsweringDataConfig


@pytest.mark.skipif(sys.platform == "win32", reason="Currently Windows is not supported")
def test_smoke_train_e2e(script_runner):
    script_runner.hf_train(
        task="question_answering",
        dataset="squad",
        model="prajjwal1/bert-tiny",
        cmd_args=["training.run_test_after_fit=False"],
    )


def test_smoke_predict_e2e(script_runner):
    seed_everything(0)
    y = script_runner.hf_predict(
        ['+x={context: "Lightning is great", question: "What is great?"}'],
        task="question_answering",
        model="prajjwal1/bert-tiny",
    )
    assert isinstance(y, dict)
    assert isinstance(y["answer"], str)


def test_model_has_correct_cfg():
    model = QuestionAnsweringTransformer(HFBackboneConfig(pretrained_model_name_or_path="bert-base-cased"))
    assert model.hparams.downstream_model_type == "transformers.AutoModelForQuestionAnswering"


def test_datamodule_has_correct_cfg():
    tokenizer = MagicMock()
    dm = QuestionAnsweringDataModule(tokenizer)
    assert type(dm.cfg) is QuestionAnsweringDataConfig
    assert dm.tokenizer is tokenizer
