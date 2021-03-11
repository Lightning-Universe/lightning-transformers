import sys

import pytest
from pytorch_lightning import seed_everything


@pytest.mark.skipif(sys.platform == "win32", reason="Currently Windows is not supported")
def test_smoke_train_e2e(script_runner):
    script_runner.hf_train(task='question_answering', dataset='squad', model='prajjwal1/bert-tiny')


def test_smoke_predict_e2e(script_runner):
    seed_everything(0)
    y = script_runner.hf_predict(
        ['+x={context: "Lightning is great", question: "What is great?"}'],
        task='question_answering',
        model='prajjwal1/bert-tiny',
    )
    assert isinstance(y, dict)
    assert 'Lightning' in y['answer']
