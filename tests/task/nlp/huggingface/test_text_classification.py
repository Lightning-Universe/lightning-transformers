import sys

import pytest


@pytest.mark.skipif(sys.platform == "win32", reason="Currently Windows is not supported")
def test_smoke_train_e2e(script_runner):
    script_runner.hf_train(task='text_classification', dataset='emotion', model='prajjwal1/bert-tiny')


def test_smoke_predict_e2e(script_runner):
    y = script_runner.hf_predict(['+x="Lightning rocks!"'], task='text_classification', model='prajjwal1/bert-tiny')
    assert len(y) == 1
    assert isinstance(y[0]['score'], float)
