# lightning-transformers

**Simple interface for high performance research using SOTA Transformers backed by [Pytorch Lightning](https://github.com/PyTorchLightning/pytorch-lightning), [Transformers](https://github.com/huggingface/transformers), and [Hydra](https://github.com/facebookresearch/hydra).**

---

<p align="center">
  <a href="#what-is-lightning-transformers">What is Lightning Transfomers</a> •
  <a href="#using-lightning-transformers">Using Lightning Transformers</a> •
  <a href="https://pytorch-lightning.readthedocs.io/transformers/">Docs</a> •
  <a href="#community">Community</a> •
  <a href="#license">License</a>
</p>

---

[![Deploy Docs](https://github.com/PyTorchLightning/lightning-transformers/actions/workflows/docs-deploy.yml/badge.svg)](https://fuzzy-disco-b18c78db.pages.github.io/)

Please note this library is in development and documentation is incomplete.

## Installation

```bash
pip install lightning-transformers
```

<details>
  <summary>Other installations</summary>

Install bleeding-edge:

```bash
pip install git+https://github.com/PytorchLightning/lightning-transformers.git@master --upgrade
```

Install all optional dependencies as well:

```bash
pip install lightning-transformers["extra"]
```

</details>

---

## What is Lightning-Transformers
Lightning transformers offers a flexible interface for training, finetuning and creating SOTA transfomer models using the PyTorch Lightning high performnace trainer.

The library includes a collection of tasks you can use for:
* Training
* Finetuning
* Inference
* Building your own Transfomer-based model

[Need to make this crystal clear- who the target audience is, what is the problem this library solves, what are the key features]

The Lightning-transfomers tasks allow you to train models using HuggingFace Transformer models and datasets, use Hydra to hotswap models, optimizers or schedulers and leverage all the advances features that Lightning has to offer, inculding custom Callbacks, Loggers, Accelerators and high performance scaling with minimal changes.

[TODO: add a gif showing how to use transformers, maybe use https://asciinema.org/, https://github.com/NickeManarin/ScreenToGif/]

## Using Lightning-Transformers

### Quick recipes

Train bert-base-cased on CARER emotion dataset using text classification task.
```bash
pl-transformers-train \
    +task=nlp/huggingface/text_classification \
    +dataset=nlp/text_classification/emotion
```

Train roberta-base backbone, on SWAG dataset multiple choice task.
```bash
pl-transformers-train \
    +task=nlp/huggingface/multiple_choice \
    +dataset=nlp/multiple_choice/swag \
    backbone.pretrained_model_name_or_path=roberta-base
```

Inference with pre-trained bert-base-cased on SQuAD using question-answering task with 2 GPUs.
```bash
pl-transformers-train \
    +task=nlp/huggingface/question_answering \
    +dataset=nlp/question_answering/squad \
    trainer.gpus=2 \
    training.do_train=False
```

Enable Sharded Training.
```bash
pl-transformers-train \
    +task=nlp/huggingface/text_classification \
    +dataset=nlp/text_classification/emotion \
    trainer=sharded
```

Enable DeepSpeed ZeRO-Offload Training.
```bash
pl-transformers-train \
    +task=nlp/huggingface/text_classification \
    +dataset=nlp/text_classification/emotion \
    trainer=zero_offload
```

### Complete API

[TODO: add here all the scripts and optional flags in a list/table]

## Contibute

Pull requests are welcome. For major changes, please open an issue first to discuss what you would like to change.

Please make sure to update tests as appropriate.

## Community
For help or questions, join our huge community on [Slack](https://join.slack.com/t/pytorch-lightning/shared_invite/zt-f6bl2l0l-JYMK3tbAgAmGRrlNr00f1A)!

## License
Please observe the Apache 2.0 license that is listed in this repository. In addition
the Lightning framework is Patent Pending.
