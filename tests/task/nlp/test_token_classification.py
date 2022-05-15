import sys
from unittest.mock import MagicMock

import pytest
import pytorch_lightning as pl
from transformers import AutoTokenizer

from lightning_transformers.task.nlp.token_classification import (
    TokenClassificationDataModule,
    TokenClassificationTransformer,
)
from lightning_transformers.task.nlp.token_classification.config import TokenClassificationDataConfig


@pytest.mark.skipif(sys.platform == "win32", reason="Currently Windows is not supported")
def test_smoke_train(hf_cache_path):
    tokenizer = AutoTokenizer.from_pretrained(pretrained_model_name_or_path="prajjwal1/bert-tiny")
    dm = TokenClassificationDataModule(
        cfg=TokenClassificationDataConfig(
            batch_size=1,
            task_name="ner",
            dataset_name="conll2003",
            preprocessing_num_workers=1,
            label_all_tokens=False,
            revision="master",
            limit_test_samples=64,
            limit_val_samples=64,
            limit_train_samples=64,
            cache_dir=hf_cache_path,
        ),
        tokenizer=tokenizer,
    )
    dm.setup("fit")
    model = TokenClassificationTransformer(pretrained_model_name_or_path="prajjwal1/bert-tiny", labels=dm.labels)

    trainer = pl.Trainer(fast_dev_run=True)
    trainer.fit(model, dm)


@pytest.mark.skipif(sys.platform == "win32", reason="Currently Windows is not supported")
def test_smoke_predict():
    model = TokenClassificationTransformer(
        pretrained_model_name_or_path="prajjwal1/bert-tiny",
        tokenizer=AutoTokenizer.from_pretrained(pretrained_model_name_or_path="prajjwal1/bert-tiny"),
        labels=2,
    )

    y = model.hf_predict("Have a good day!")
    assert len(y) == 5
    assert [a["word"] for a in y] == ["have", "a", "good", "day", "!"]


def test_model_has_correct_cfg():
    model = TokenClassificationTransformer(
        pretrained_model_name_or_path="bert-base-cased",
        labels=2,
    )
    assert model.hparams.downstream_model_type == "transformers.AutoModelForTokenClassification"
    assert model.num_labels == 2


def test_datamodule_has_correct_cfg():
    tokenizer = MagicMock()
    dm = TokenClassificationDataModule(tokenizer)
    assert type(dm.cfg) is TokenClassificationDataConfig
    assert dm.tokenizer is tokenizer
