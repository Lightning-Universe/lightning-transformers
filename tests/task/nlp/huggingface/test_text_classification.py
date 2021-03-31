import sys
from unittest.mock import MagicMock

import pytest

from lightning_transformers.core.nlp.huggingface import HFBackboneConfig, HFTransformerDataConfig
from lightning_transformers.task.nlp.text_classification import (
    TextClassificationDataModule,
    TextClassificationTransformer,
)


@pytest.mark.skipif(sys.platform == "win32", reason="Currently Windows is not supported")
def test_smoke_train_e2e(script_runner):
    script_runner.hf_train(task='text_classification', dataset='emotion', model='prajjwal1/bert-tiny')


def test_smoke_predict_e2e(script_runner):
    y = script_runner.hf_predict(['+x="Lightning rocks!"'], task='text_classification', model='prajjwal1/bert-tiny')
    assert len(y) == 1
    assert isinstance(y[0]['score'], float)


def test_predict_from_ckpt_path(script_runner, tmpdir):
    script_runner.hf_train(
        task='text_classification',
        dataset='emotion',
        model='prajjwal1/bert-tiny',
        cmd_args=[f'trainer.default_root_dir={tmpdir}'],
        fast_dev_run=0
    )
    ckpt_path = tmpdir / 'checkpoints' / 'epoch=0-step=0.ckpt'
    assert ckpt_path.exists()

    y = script_runner.hf_predict(
        ['+x="Lightning rocks!"', f'+checkpoint_path="{ckpt_path}"'],
        task='text_classification',
        model='prajjwal1/bert-tiny',
    )
    assert len(y) == 1
    assert isinstance(y[0]['score'], float)


def test_model_can_be_created():
    TextClassificationTransformer(
        'transformers.AutoModelForTokenClassification',
        HFBackboneConfig(pretrained_model_name_or_path='bert-base-cased'),
    )


def test_datamodule_has_correct_cfg():
    tokenizer = MagicMock()
    dm = TextClassificationDataModule(tokenizer)
    assert type(dm.cfg) is HFTransformerDataConfig
    assert dm.tokenizer is tokenizer
