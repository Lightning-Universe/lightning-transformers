# lightning-transformers

**Lightning Transformers provides capabilities for high performance research using SOTA Transformers backed by [Pytorch Lightning](https://github.com/PyTorchLightning/pytorch-lightning), [Transformers](https://github.com/huggingface/transformers), and [Hydra](https://github.com/facebookresearch/hydra).**

---
[![Deploy Docs](https://github.com/PyTorchLightning/lightning-transformers/actions/workflows/docs-deploy.yml/badge.svg)](https://fuzzy-disco-b18c78db.pages.github.io/)

Please note this library is in development and documentation is incomplete.

## Installation

```bash
pip install lightning-transformers
```

Install bleeding-edge:

```bash
pip install git+https://github.com/PytorchLightning/lightning-transformers.git@master --upgrade
```

Install all optional dependencies as well:

```bash
pip install lightning-transformers["extra"]
```

## QuickStart

Train bert-base-cased on CARER emotion dataset using text classification task.
```bash
python pl-transformers-train \
    +task=nlp/huggingface/text_classification \
    +dataset=nlp/text_classification/emotion
```

Train roberta-base backbone, on SWAG dataset multiple choice task.
```bash
python pl-transformers-train \
    +task=nlp/huggingface/multiple_choice \
    +dataset=nlp/multiple_choice/swag \
    backbone.pretrained_model_name_or_path=roberta-base
```

Inference with pre-trained bert-base-cased on SQuAD using question-answering task with 2 GPUs.
```bash
python pl-transformers-train \
    +task=nlp/huggingface/question_answering \
    +dataset=nlp/question_answering/squad \
    trainer.gpus=2 \
    training.do_train=False
```

Enable Sharded Training.
```bash
python pl-transformers-train \
    +task=nlp/huggingface/text_classification \
    +dataset=nlp/text_classification/emotion \
    trainer=sharded
```

Enable DeepSpeed ZeRO-Offload Training.
```bash
python pl-transformers-train \
    +task=nlp/huggingface/text_classification \
    +dataset=nlp/text_classification/emotion \
    trainer=zero_offload
```
