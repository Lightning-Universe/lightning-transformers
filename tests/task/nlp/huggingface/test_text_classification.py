import sys

import pytest


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
