# lightning-transformers

The lightweight PyTorch wrapper for high-performance AI research.
Scale your models, not the boilerplate.

Transformers provides thousands of pretrained models to perform tasks on texts such as classification, information extraction, question answering, summarization, translation, text generation, etc in 100+ languages

Lightning-Transformers disentangles Transformers code to decouple the science from the engineering. 

### Current API
```bash
# Train bert-base-cased on CARER emotion dataset using text classification task provided by huggingface
python train.py \
    +task=huggingface/text_classification \
    +dataset=text_classification/emotion 

# Train roberta-base backbone, on SWAG dataset multiple choice task provided by huggingface
python train.py \
    +task=huggingface/multiple_choice \
    +dataset=multiple_choice/swag \
    backbone=roberta-base

# Train bert-base-cased on SQuAD using question-answering task provided by huggingface with 1 gpu and batch_size=4
python train.py \
    +task=huggingface/question_answering \
    +dataset=question_answering/squad
    trainer.gpus=1 \
    training.batch_size=4

# Make an inference with pre-trained bert-base-cased on SQuAD using question-answering task provided by huggingface with 2 gpu.
python train.py \
    +task=huggingface/question_answering \
    +dataset=question_answering/squad \
    trainer.gpus=2 \
    training.do_train=False

# Enable DDP + Sharding with 2 GPUs
python train.py \
    +task=huggingface/text_classification \
    +dataset=text_classification/emotion \
    trainer=sharded

# Swap to a different huggingface transformer model with DDP + Sharding with 2 GPUs
python train.py \
    +task=huggingface/text_classification \
    +dataset=text_classification/emotion \
    backbone.pretrained_model_name_or_path=roberta-base \
    trainer=sharded

# (An example, not real) 
# Swap to fairseq model that is compatible with text classification task provided by huggingface
# Model should be defined in the conf/model/fairseq.yaml
python train.py \
    +task=huggingface/text_classification \
    +dataset=text_classification/emotion \
    +backbone=fairseq

WIP

# Train bert-base-cased on CARER emotion dataset using text classification task provided by huggingface
python train.py \
    +task=huggingface/multiple_choice \
    +dataset=multiple_choice/swag
```