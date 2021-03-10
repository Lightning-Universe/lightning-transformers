import sys

import pytest


@pytest.mark.skipif(sys.platform == "win32", reason="Currently Windows is not supported")
def test_smoke_train_e2e(script_runner):
    script_runner.hf_train(task='translation', dataset='wmt16', model='patrickvonplaten/t5-tiny-random')


def test_smoke_predict_e2e(script_runner):
    y = script_runner.hf_predict(['+x="¡Hola Sean!"'], task='translation', model='patrickvonplaten/t5-tiny-random')
    assert len(y) == 1
    assert isinstance(y[0]['translation_text'], str)
