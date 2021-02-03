def test_language_model(hydra_runner):
    task = 'nlp/huggingface/language_modeling'
    dataset = 'nlp/language_modeling/wikitext'
    suffix = 'backbone.pretrained_model_name_or_path=prajjwal1/bert-tiny'
    hydra_runner(task, dataset, suffix)


def test_multiple_choice(hydra_runner):
    task = 'nlp/huggingface/multiple_choice'
    dataset = 'nlp/multiple_choice/race'
    suffix = 'backbone.pretrained_model_name_or_path=prajjwal1/bert-tiny'
    hydra_runner(task, dataset, suffix)


# def test_question_answering(hydra_runner):
#     task = 'nlp/huggingface/question_answering'
#     dataset = 'nlp/question_answering/squad'
#     suffix = 'backbone.pretrained_model_name_or_path=prajjwal1/bert-tiny'
#     hydra_runner(task, dataset, suffix)

# def test_summarization(hydra_runner):
#     task = 'nlp/huggingface/summarization'
#     dataset = 'nlp/summarization/xsum'
#     suffix = 'backbone.pretrained_model_name_or_path=sshleifer/tiny-mbart'
#     hydra_runner(task, dataset, suffix)


def test_token_classification(hydra_runner):
    task = 'nlp/huggingface/token_classification'
    dataset = 'nlp/token_classification/conll'
    suffix = 'backbone.pretrained_model_name_or_path=prajjwal1/bert-tiny'
    hydra_runner(task, dataset, suffix)


def test_text_classification(hydra_runner):
    task = 'nlp/huggingface/text_classification'
    dataset = 'nlp/text_classification/emotion'
    suffix = 'backbone.pretrained_model_name_or_path=prajjwal1/bert-tiny'
    hydra_runner(task, dataset, suffix)


def translation(hydra_runner):
    task = 'nlp/huggingface/translation'
    dataset = 'nlp/translation/wmt16'
    suffix = 'backbone.pretrained_model_name_or_path=sshleifer/tiny-mbart'
    hydra_runner(task, dataset, suffix)
