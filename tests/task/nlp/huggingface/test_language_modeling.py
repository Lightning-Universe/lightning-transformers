import sys

import pytest


@pytest.mark.skipif(sys.platform == "win32", reason="Currently Windows is not supported")
def test_smoke_train_e2e(script_runner):
    script_runner.hf_train(task='language_modeling', dataset='wikitext', model='prajjwal1/bert-tiny')


def test_smoke_predict_e2e(script_runner):
    # y = script_runner.hf_predict(['+x="The [MASK] house"'], task='language_modeling', model='prajjwal1/bert-tiny')
    # TODO: PipelineException: The model 'BertLMHeadModel' is not supported for text-generation
    pass
