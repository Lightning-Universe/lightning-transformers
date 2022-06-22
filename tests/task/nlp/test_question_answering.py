import sys
from unittest.mock import MagicMock

import pytest
import pytorch_lightning as pl
import transformers
from transformers import AutoTokenizer

from lightning_transformers.task.nlp.question_answering import (
    QuestionAnsweringDataModule,
    QuestionAnsweringTransformer,
    SquadDataModule,
)


@pytest.mark.skipif(sys.platform == "win32", reason="Currently Windows is not supported")
@pytest.mark.skip(reason="Currently Question Answering is broken.")
def test_smoke_train(hf_cache_path):
    tokenizer = AutoTokenizer.from_pretrained(pretrained_model_name_or_path="sshleifer/tiny-gpt2")
    model = QuestionAnsweringTransformer(pretrained_model_name_or_path="sshleifer/tiny-gpt2")
    dm = SquadDataModule(
        batch_size=1,
        dataset_config_name="plain_text",
        max_length=384,
        version_2_with_negative=False,
        null_score_diff_threshold=0.0,
        doc_stride=128,
        n_best_size=20,
        max_answer_length=30,
        limit_train_samples=64,
        limit_test_samples=64,
        limit_val_samples=64,
        cache_dir=hf_cache_path,
        tokenizer=tokenizer,
    )
    trainer = pl.Trainer(fast_dev_run=True)

    trainer.fit(model, dm)


@pytest.mark.skipif(sys.platform == "win32", reason="Currently Windows is not supported")
@pytest.mark.skip(reason="Currently Question Answering is broken.")
def test_smoke_predict():
    model = QuestionAnsweringTransformer(
        pretrained_model_name_or_path="sshleifer/tiny-gpt2",
        tokenizer=AutoTokenizer.from_pretrained(pretrained_model_name_or_path="sshleifer/tiny-gpt2"),
    )
    y = model.hf_predict(dict(context="Lightning is great", question="What is great?"))
    assert isinstance(y, dict)
    assert isinstance(y["answer"], str)


def test_model_has_correct_cfg():
    model = QuestionAnsweringTransformer(pretrained_model_name_or_path="bert-base-cased")
    assert isinstance(model.model, transformers.BertForQuestionAnswering)


def test_datamodule_has_tokenizer():
    tokenizer = MagicMock()
    dm = QuestionAnsweringDataModule(tokenizer)
    assert dm.tokenizer is tokenizer
