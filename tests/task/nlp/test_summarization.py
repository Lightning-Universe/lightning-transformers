import sys
from unittest.mock import MagicMock

import pytest
import pytorch_lightning as pl
from transformers import AutoTokenizer

from lightning_transformers.task.nlp.summarization import (
    SummarizationConfig,
    SummarizationDataConfig,
    SummarizationDataModule,
    SummarizationTransformer,
    XsumSummarizationDataModule,
)


@pytest.mark.skipif(sys.platform == "win32", reason="Currently Windows is not supported")
def test_smoke_train(hf_cache_path):
    tokenizer = AutoTokenizer.from_pretrained(pretrained_model_name_or_path="patrickvonplaten/t5-tiny-random")
    model = SummarizationTransformer(
        pretrained_model_name_or_path="patrickvonplaten/t5-tiny-random",
        cfg=SummarizationConfig(
            use_stemmer=True,
            val_target_max_length=142,
            num_beams=None,
            compute_generate_metrics=True,
        ),
    )
    dm = XsumSummarizationDataModule(
        cfg=SummarizationDataConfig(
            limit_train_samples=64,
            limit_val_samples=64,
            limit_test_samples=64,
            batch_size=1,
            dataset_name="xsum",
            max_source_length=128,
            max_target_length=128,
            cache_dir=hf_cache_path,
        ),
        tokenizer=tokenizer,
    )
    trainer = pl.Trainer(fast_dev_run=True)

    trainer.fit(model, dm)


@pytest.mark.skipif(sys.platform == "win32", reason="Currently Windows is not supported")
def test_smoke_predict():
    model = SummarizationTransformer(
        pretrained_model_name_or_path="patrickvonplaten/t5-tiny-random",
        tokenizer=AutoTokenizer.from_pretrained(pretrained_model_name_or_path="patrickvonplaten/t5-tiny-random"),
    )

    y = model.hf_predict(
        "The results found significant improvements over all tasks evaluated",
        min_length=2,
        max_length=12,
    )
    assert len(y) == 1
    output = y[0]["summary_text"]
    assert 2 <= len(output.split()) <= 12


def test_model_has_correct_cfg():
    model = SummarizationTransformer(pretrained_model_name_or_path="t5-base")
    assert model.hparams.downstream_model_type == "transformers.AutoModelForSeq2SeqLM"
    assert type(model.cfg) is SummarizationConfig


def test_datamodule_has_correct_cfg():
    tokenizer = MagicMock()
    dm = SummarizationDataModule(tokenizer)
    assert type(dm.cfg) is SummarizationDataConfig
    assert dm.tokenizer is tokenizer
