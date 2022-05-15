import sys
from unittest.mock import MagicMock

import pytest
import pytorch_lightning as pl
from transformers import AutoTokenizer

from lightning_transformers.task.nlp.multiple_choice import (
    MultipleChoiceDataConfig,
    MultipleChoiceDataModule,
    MultipleChoiceTransformer,
    SwagMultipleChoiceDataModule,
)


@pytest.mark.skipif(sys.platform == "win32", reason="Currently Windows is not supported")
def test_smoke_train(hf_cache_path):
    tokenizer = AutoTokenizer.from_pretrained(pretrained_model_name_or_path="prajjwal1/bert-tiny")
    model = MultipleChoiceTransformer(pretrained_model_name_or_path="prajjwal1/bert-tiny")
    dm = SwagMultipleChoiceDataModule(
        cfg=MultipleChoiceDataConfig(
            batch_size=1,
            dataset_name="swag",
            dataset_config_name="regular",
            padding=False,
            cache_dir=hf_cache_path,
        ),
        tokenizer=tokenizer,
    )
    trainer = pl.Trainer(fast_dev_run=True)

    trainer.fit(model, dm)


def test_model_has_correct_cfg():
    model = MultipleChoiceTransformer(pretrained_model_name_or_path="bert-base-cased")
    assert model.hparams.downstream_model_type == "transformers.AutoModelForMultipleChoice"


def test_datamodule_has_correct_cfg():
    tokenizer = MagicMock()
    dm = MultipleChoiceDataModule(tokenizer)
    assert isinstance(dm.cfg, MultipleChoiceDataConfig)
    assert dm.tokenizer is tokenizer
