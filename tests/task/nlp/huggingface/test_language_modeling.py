import sys

import pytest


@pytest.mark.skipif(sys.platform == "win32", reason="Currently Windows is not supported")
def test_smoke_train_e2e(script_runner):
    script_runner.hf_train(task='language_modeling', dataset='wikitext', model='prajjwal1/bert-tiny')


def test_smoke_predict_e2e(script_runner):
    y = script_runner.hf_predict(['+x="The house:"'], task='language_modeling', model='sshleifer/tiny-gpt2')
    assert len(y) == 1
    assert isinstance(y[0]['generated_text'], str)
