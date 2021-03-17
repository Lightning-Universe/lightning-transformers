# lightning-transformers

**Flexible interface for high performance research using SOTA Transformers leveraging [Pytorch Lightning](https://github.com/PyTorchLightning/pytorch-lightning), [Transformers](https://github.com/huggingface/transformers), and [Hydra](https://github.com/facebookresearch/hydra).**

---

<p align="center">
  <a href="#what-is-lightning-transformers">What is Lightning Transfomers</a> •
  <a href="#using-lightning-transformers">Using Lightning Transformers</a> •
  <a href="https://pytorch-lightning.readthedocs.io/transformers/">Docs</a> •
  <a href="#community">Community</a> •
  <a href="#license">License</a>
</p>

---

[![Temporary Docs](https://github.com/PyTorchLightning/lightning-transformers/actions/workflows/docs-deploy.yml/badge.svg)](https://fuzzy-disco-b18c78db.pages.github.io/)

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

Lightning Transformers offers a flexible interface for training and fine-tuning SOTA Transformer models using the PyTorch Lightning Trainer.

### Why Lightning Transformers?

* **Powered by PyTorch Lightning** - Leverage everything that `PyTorch Lightning <https://github.com/PyTorchLightning/pytorch-lightning>`_ has to offer, allowing you to use Lightning provided and custom Callbacks, Loggers, Accelerators and high performance scaling with minimal changes.
* **Backed by HuggingFace Transformers** - Train using `HuggingFace Transformer <https://github.com/huggingface/transformers>`_ models and datasets, across the expansive library spanning multiple modalities and tasks within NLP/Audio and Vision.
* **Transformer Task Abstraction for Rapid Research & Experimentation** - Built from the ground up to be task agnostic, the library supports creating transformer tasks across all modalities with little friction.
* **Powerful config composition backed by Hydra** - Leverage the config structure to swap out models, optimizers, schedulers task and many more configurations without touching the code.
* **Seamless Memory and Speed Optimizations** - We provide seamless integration to enable training optimizations, such as `DeepSpeed ZeRO <https://pytorch-lightning.readthedocs.io/en/latest/multi_gpu.html#deepspeed>`_ or `FairScale Sharded Training <https://pytorch-lightning.readthedocs.io/en/latest/multi_gpu.html#sharded-training>`_ with no code changes.

Lightning Transformers tasks allow you to train models using HuggingFace Transformer models and datasets, use Hydra to hotswap models, optimizers or schedulers and leverage all the advances features that Lightning has to offer, inculding custom Callbacks, Loggers, Accelerators and high performance scaling with minimal changes.

[TODO: add a gif showing how to use transformers, maybe use https://asciinema.org/, https://github.com/NickeManarin/ScreenToGif/]

## Using Lightning-Transformers

### Quick recipes

Train bert-base-cased on CARER emotion dataset using text classification task.
```bash
python train.py \
    +task=nlp/huggingface/text_classification \
    +dataset=nlp/text_classification/emotion
```

Train roberta-base backbone, on SWAG dataset multiple choice task.
```bash
python train.py \
    +task=nlp/huggingface/multiple_choice \
    +dataset=nlp/multiple_choice/swag \
    backbone.pretrained_model_name_or_path=roberta-base
```

Inference with pre-trained bert-base-cased on SQuAD using question-answering task with 2 GPUs.
```bash
python train.py \
    +task=nlp/huggingface/question_answering \
    +dataset=nlp/question_answering/squad \
    trainer.gpus=2 \
    training.do_train=False
```

Enable Sharded Training.
```bash
python train.py \
    +task=nlp/huggingface/text_classification \
    +dataset=nlp/text_classification/emotion \
    trainer=sharded
```

Enable DeepSpeed ZeRO-Offload Training.
```bash
python train.py \
    +task=nlp/huggingface/text_classification \
    +dataset=nlp/text_classification/emotion \
    trainer=zero_offload
```

### Complete API

[TODO: add here all the scripts and optional flags in a list/table]

## Contribute

Pull requests are welcome. For major changes, please open an issue first to discuss what you would like to change.

Please make sure to update tests as appropriate.

## Community
For help or questions, join our huge community on [Slack](https://join.slack.com/t/pytorch-lightning/shared_invite/zt-f6bl2l0l-JYMK3tbAgAmGRrlNr00f1A)!

## License
Please observe the Apache 2.0 license that is listed in this repository. In addition, the Lightning framework is Patent Pending.
