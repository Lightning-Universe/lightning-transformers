def run_hf_hydra_runner(hydra_runner, task, dataset, model, max_samples=64):
    suffix = f'backbone.pretrained_model_name_or_path={model} dataset.cfg.max_samples={max_samples}'
    hydra_runner(task=f'nlp/huggingface/{task}', dataset=f'nlp/{task}/{dataset}', suffix=suffix)


def test_language_model(hydra_runner):
    run_hf_hydra_runner(
        hydra_runner=hydra_runner, task='language_modeling', dataset='wikitext', model='prajjwal1/bert-tiny'
    )


def test_question_answering(hydra_runner):
    run_hf_hydra_runner(
        hydra_runner=hydra_runner, task='question_answering', dataset='squad', model='prajjwal1/bert-tiny'
    )


def test_summarization(hydra_runner):
    run_hf_hydra_runner(
        hydra_runner=hydra_runner, task='summarization', dataset='xsum', model='patrickvonplaten/t5-tiny-random'
    )


def test_multiple_choice(hydra_runner):
    run_hf_hydra_runner(hydra_runner=hydra_runner, task='multiple_choice', dataset='race', model='prajjwal1/bert-tiny')


def test_token_classification(hydra_runner):
    run_hf_hydra_runner(
        hydra_runner=hydra_runner, task='token_classification', dataset='conll', model='prajjwal1/bert-tiny'
    )


def test_text_classification(hydra_runner):
    run_hf_hydra_runner(
        hydra_runner=hydra_runner, task='text_classification', dataset='emotion', model='prajjwal1/bert-tiny'
    )


def test_translation(hydra_runner):
    run_hf_hydra_runner(
        hydra_runner=hydra_runner, task='translation', dataset='wmt16', model='patrickvonplaten/t5-tiny-random'
    )
