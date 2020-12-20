# lightning-transformers

### Current API
```bash
# Train bert-base-cased on CARER emotion dataset using text classification task provided by huggingface
python train.py
    +task/text_classification=huggingface \
    +dataset/text_classification=emotion

# Train bert-base-cased on SQuAD using question-answering task provided by huggingface
python train.py
    +task/question_answering=huggingface \
    +dataset/question_answering=squad

# Swap to a different huggingface transformer model
python train.py
    +task/text_classification=huggingface \
    +dataset/text_classification=emotion \
    +model.pretrained_model_name_or_path=roberta-base

# Enable DDP + Sharding
python train.py
    +task/text_classification=huggingface \
    +dataset/text_classification=emotion \
    +model.pretrained_model_name_or_path=roberta-base \
    trainer.gpus=4 \
    +trainer.accelerator=ddp \
    +trainer.plugins=ddp_sharded

# (An example, not real) 
# Swap to fairseq model that is compatible with text classification task provided by huggingface
# Model should be defined in the conf/model/fairseq.yaml
python train.py
    +task/text_classification=huggingface \
    +dataset/text_classification=emotion \
    +model=fairseq
```