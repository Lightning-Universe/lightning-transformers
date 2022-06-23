import os
import sys
from unittest.mock import MagicMock

import pytest
import pytorch_lightning as pl
import torch
import transformers
from transformers import AutoTokenizer

from lightning_transformers.task.nlp.language_modeling import LanguageModelingDataModule, LanguageModelingTransformer
from lightning_transformers.utilities.imports import _ACCELERATE_AVAILABLE

if _ACCELERATE_AVAILABLE:
    from accelerate import init_empty_weights


@pytest.mark.skipif(sys.platform == "win32", reason="Currently Windows is not supported")
@pytest.mark.parametrize("streaming", [True, False])
def test_smoke_train(hf_cache_path, streaming):
    tokenizer = AutoTokenizer.from_pretrained(pretrained_model_name_or_path="prajjwal1/bert-tiny")
    model = LanguageModelingTransformer(pretrained_model_name_or_path="prajjwal1/bert-tiny")
    dm = LanguageModelingDataModule(
        batch_size=1,
        dataset_name="wikitext",
        dataset_config_name="wikitext-2-raw-v1",
        cache_dir=hf_cache_path,
        streaming=streaming,
        tokenizer=tokenizer,
    )
    trainer = pl.Trainer(max_steps=10, max_epochs=1, limit_val_batches=1, limit_test_batches=1, limit_train_batches=1)

    trainer.fit(model, dm)


@pytest.mark.skipif(sys.platform == "win32", reason="Currently Windows is not supported")
def test_smoke_predict():
    model = LanguageModelingTransformer(
        pretrained_model_name_or_path="prajjwal1/bert-tiny",
        tokenizer=AutoTokenizer.from_pretrained(pretrained_model_name_or_path="prajjwal1/bert-tiny"),
    )
    y = model.hf_predict("The house:")
    assert len(y) == 1
    assert y[0]["generated_text"].startswith("The house:")


def test_model_has_correct_cfg():
    model = LanguageModelingTransformer(pretrained_model_name_or_path="prajjwal1/bert-tiny")
    assert isinstance(model.model, transformers.BertLMHeadModel)


def test_datamodule_has_tokenizer():
    tokenizer = MagicMock()
    dm = LanguageModelingDataModule(tokenizer)
    assert dm.tokenizer is tokenizer


@pytest.mark.skipif(not _ACCELERATE_AVAILABLE, reason="Accelerate not installed.")
@pytest.mark.skipif(not torch.cuda.is_available(), reason="Requires a GPU to run.")
def test_generate_inference(tmpdir):
    model = LanguageModelingTransformer(
        pretrained_model_name_or_path="sshleifer/tiny-gpt2",
        tokenizer=AutoTokenizer.from_pretrained("sshleifer/tiny-gpt2"),
    )
    ckpt_path = os.path.join(tmpdir, "checkpoint.ckpt")
    torch.save(model.model.state_dict(), ckpt_path)

    with init_empty_weights():
        model = LanguageModelingTransformer(
            pretrained_model_name_or_path="sshleifer/tiny-gpt2",
            tokenizer=AutoTokenizer.from_pretrained("sshleifer/tiny-gpt2"),
            load_weights=False,
        )

    model.load_checkpoint_and_dispatch(ckpt_path, device_map="auto")

    output = model.generate("Hello, my name is", device=torch.device("cuda"))
    output = model.tokenizer.decode(output[0].tolist())
    assert "Hello, my name is" in output
