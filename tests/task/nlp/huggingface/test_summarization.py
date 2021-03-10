import sys

import pytest


@pytest.mark.skipif(sys.platform == "win32", reason="Currently Windows is not supported")
def test_smoke_train_e2e(script_runner):
    script_runner.hf_train(task='summarization', dataset='xsum', model='patrickvonplaten/t5-tiny-random')


def test_smoke_predict_e2e(script_runner):
    y = script_runner.hf_predict(
        [
            '+x="The results found significant improvements over all tasks evaluated"',
            '+predict_kwargs={min_length: 2, max_length: 12}',
        ],
        task='summarization',
        model='patrickvonplaten/t5-tiny-random',
    )
    assert len(y) == 1
    output = y[0]['summary_text']
    assert 2 <= len(output.split()) <= 12
