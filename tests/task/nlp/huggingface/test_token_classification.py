import sys

import pytest


@pytest.mark.skipif(sys.platform == "win32", reason="Currently Windows is not supported")
def test_smoke_train_e2e(script_runner):
    script_runner.hf_train(task='token_classification', dataset='conll', model='prajjwal1/bert-tiny')


def test_smoke_predict_e2e(script_runner):
    y = script_runner.hf_predict(
        ['+x="Have a good day!"', '+model_data_args={labels: 2}'],
        task='token_classification',
        model='prajjwal1/bert-tiny',
    )
    assert len(y) == 5
    assert [a['word'] for a in y] == ['have', 'a', 'good', 'day', '!']
