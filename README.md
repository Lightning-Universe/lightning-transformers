# lightning-transformers

### Current API
```bash
# Train bert-base-cased on CARER emotion dataset using text classification task provided by huggingface
python train.py \
    +task=huggingface/text_classification \
    +dataset=text_classification/emotion \

# Train bert-base-cased on SQuAD using question-answering task provided by huggingface with 1 gpu and batch_size=4
python train.py \
    +task=huggingface/question_answering \
    +dataset=question_answering/squad
    trainer.gpus=1 \
    training.batch_size=4

# Make an inference with pre-trained bert-base-cased on SQuAD using question-answering task provided by huggingface with 2 gpu.
python train.py     \
    +task=huggingface/question_answering \
    +dataset=question_answering/squad \
    trainer.gpus=2 \
    training.do_train=False

# Enable DDP + Sharding
python train.py \
    +task=huggingface/text_classification \
    +dataset=text_classification/emotion \
    trainer=ddp_shared_2_gpus

# Swap to a different huggingface transformer model with DDP + Sharding
python train.py \
    +task=huggingface/text_classification \
    +dataset=text_classification/emotion \
    model.pretrained_model_name_or_path=roberta-base \
    trainer=ddp_shared_2_gpus \

# (An example, not real) 
# Swap to fairseq model that is compatible with text classification task provided by huggingface
# Model should be defined in the conf/model/fairseq.yaml
python train.py \
    +task=huggingface/text_classification \
    +dataset=text_classification/emotion \
    +model=fairseq
```