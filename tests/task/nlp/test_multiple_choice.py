import sys
from unittest.mock import MagicMock

import pytest
import pytorch_lightning as pl
import transformers
from transformers import AutoTokenizer

from lightning_transformers.task.nlp.multiple_choice import (
    MultipleChoiceDataModule,
    MultipleChoiceTransformer,
    SwagMultipleChoiceDataModule,
)


@pytest.mark.skipif(sys.platform == "win32", reason="Currently Windows is not supported")
def test_smoke_train(hf_cache_path):
    tokenizer = AutoTokenizer.from_pretrained(pretrained_model_name_or_path="prajjwal1/bert-tiny")
    model = MultipleChoiceTransformer(pretrained_model_name_or_path="prajjwal1/bert-tiny")
    dm = SwagMultipleChoiceDataModule(
        batch_size=1,
        dataset_config_name="regular",
        padding=False,
        cache_dir=hf_cache_path,
        tokenizer=tokenizer,
    )
    trainer = pl.Trainer(fast_dev_run=True)

    trainer.fit(model, dm)


def test_model_has_correct_cfg():
    model = MultipleChoiceTransformer(pretrained_model_name_or_path="bert-base-cased")
    assert isinstance(model.model, transformers.BertForMultipleChoice)


def test_datamodule_has_tokenizer():
    tokenizer = MagicMock()
    dm = MultipleChoiceDataModule(tokenizer)
    assert dm.tokenizer is tokenizer
