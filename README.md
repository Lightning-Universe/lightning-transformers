
<img src="docs/source/_static/images/icon.png" width="80px">
<h1> Lightning Transformers </h1>

**Flexible interface for high performance research using SOTA Transformers leveraging [Pytorch Lightning](https://github.com/PyTorchLightning/pytorch-lightning), [Transformers](https://github.com/huggingface/transformers), and [Hydra](https://github.com/facebookresearch/hydra).**

<div align="center">
  <img src="https://pl-bolts-doc-images.s3.us-east-2.amazonaws.com/lit-tfmrs.gif">
</div>

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

Docs are still currently in progress, including this README. However to get started it's useful to take a step by step look at the docs like below:

1. [Understanding the Config Structure](https://fuzzy-disco-b18c78db.pages.github.io/structure/conf.html)
2. [Creating your own Task, Backbone and Dataset](https://fuzzy-disco-b18c78db.pages.github.io/tasks/new.html)
3. [Out of the box NLP Tasks](https://fuzzy-disco-b18c78db.pages.github.io/tasks/nlp.html)

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

Lightning Transformers offers a flexible interface for training and fine-tuning SOTA Transformer models using the [PyTorch Lightning Trainer](https://pytorch-lightning.readthedocs.io/en/stable/common/trainer.html).

* **Train [HuggingFace Transformer](https://github.com/huggingface/transformers) models and datasets** with Lightning custom Callbacks, Loggers, Accelerators and high performance scaling.
* **Seamless Memory and Speed Optimizations** such as [DeepSpeed ZeRO](https://pytorch-lightning.readthedocs.io/en/latest/multi_gpu.html#deepspeed) or [FairScale Sharded Training](https://pytorch-lightning.readthedocs.io/en/latest/multi_gpu.html#sharded-training) with no code changes.
* **Powerful config composition backed by [Hydra](https://hydra.cc/)** - Easily swap out models, optimizers, schedulers and many more configurations without touching the code.
* **Transformer Task Abstraction for Rapid Research & Experimentation** - Built from the ground up to be task agnostic, the library supports creating transformer tasks across all modalities with little friction.

Lightning Transformers tasks allow you to train models using HuggingFace Transformer models and datasets, use Hydra to hotswap models, optimizers or schedulers and leverage all the advances features that Lightning has to offer, inculding custom Callbacks, Loggers, Accelerators and high performance scaling with minimal changes.

## Using Lightning-Transformers

### Quick recipes

#### Train bert-base-cased on CARER emotion dataset using text classification task.
```bash
python train.py \
    +task=nlp/text_classification \
    +dataset=nlp/text_classification/emotion
```

<details>
  <summary>See the default config used</summary>

```python
optimizer:
  _target_: torch.optim.AdamW
  lr: ${training.lr}
  weight_decay: 0.001
scheduler:
  _target_: transformers.get_linear_schedule_with_warmup
  num_training_steps: -1
  num_warmup_steps: 0.1
training:
  do_train: true
  do_eval: true
  lr: 5.0e-05
  output_dir: .
  batch_size: 16
  num_workers: 16
trainer:
  _target_: pytorch_lightning.Trainer
  logger: true
  checkpoint_callback: true
  callbacks: null
  default_root_dir: null
  gradient_clip_val: 0.0
  process_position: 0
  num_nodes: 1
  num_processes: 1
  gpus: null
  auto_select_gpus: false
  tpu_cores: null
  log_gpu_memory: null
  progress_bar_refresh_rate: 1
  overfit_batches: 0.0
  track_grad_norm: -1
  check_val_every_n_epoch: 1
  fast_dev_run: false
  accumulate_grad_batches: 1
  max_epochs: 1
  min_epochs: 1
  max_steps: null
  min_steps: null
  limit_train_batches: 1.0
  limit_val_batches: 1.0
  limit_test_batches: 1.0
  val_check_interval: 1.0
  flush_logs_every_n_steps: 100
  log_every_n_steps: 50
  accelerator: null
  sync_batchnorm: false
  precision: 32
  weights_summary: top
  weights_save_path: null
  num_sanity_val_steps: 2
  truncated_bptt_steps: null
  resume_from_checkpoint: null
  profiler: null
  benchmark: false
  deterministic: false
  reload_dataloaders_every_epoch: false
  auto_lr_find: false
  replace_sampler_ddp: true
  terminate_on_nan: false
  auto_scale_batch_size: false
  prepare_data_per_node: true
  plugins: null
  amp_backend: native
  amp_level: O2
  move_metrics_to_cpu: false
task:
  _recursive_: false
  backbone: ${backbone}
  optimizer: ${optimizer}
  scheduler: ${scheduler}
  _target_: lightning_transformers.task.nlp..text_classification.TextClassificationTransformer
  downstream_model_type: transformers.AutoModelForSequenceClassification
dataset:
  cfg:
    batch_size: ${training.batch_size}
    num_workers: ${training.num_workers}
    dataset_name: emotion
    dataset_config_name: null
    train_file: null
    validation_file: null
    test_file: null
    train_val_split: null
    max_samples: null
    cache_dir: null
    padding: max_length
    truncation: only_first
    preprocessing_num_workers: 1
    load_from_cache_file: true
    max_length: 128
    limit_train_samples: null
    limit_val_samples: null
    limit_test_samples: null
  _target_: lightning_transformers.task.nlp.text_classification.TextClassificationDataModule
experiment_name: ${now:%Y-%m-%d}_${now:%H-%M-%S}
log: false
ignore_warnings: true
tokenizer:
  _target_: transformers.AutoTokenizer.from_pretrained
  pretrained_model_name_or_path: ${backbone.pretrained_model_name_or_path}
  use_fast: true
backbone:
  pretrained_model_name_or_path: bert-base-cased
```
</details>

#### Swap the backbone to [RoBERTa](https://huggingface.co/transformers/model_doc/roberta.html) and the optimizer to RMSprop:

```bash
python train.py \
    +task=nlp/text_classification \
    +dataset=nlp/text_classification/emotion
    backbone.pretrained_model_name_or_path=roberta-base
    optimizer=rmsprop
```

<details>
  <summary>See the changed config under-the-hood</summary>

```diff
 optimizer:
-  _target_: torch.optim.AdamW
+  _target_: torch.optim.RMSprop
   lr: ${training.lr}
-  weight_decay: 0.001
 scheduler:
   _target_: transformers.get_linear_schedule_with_warmup
   num_training_steps: -1
....
tokenizer:
   pretrained_model_name_or_path: ${backbone.pretrained_model_name_or_path}
   use_fast: true
 backbone:
-  pretrained_model_name_or_path: bert-base-cased
+  pretrained_model_name_or_path: roberta-base
```
</details>

#### Enable Sharded Training.
```bash
python train.py \
    +task=nlp/text_classification \
    +dataset=nlp/text_classification/emotion \
    trainer=sharded
```

<details>
  <summary>See the modified config</summary>
Without the need to modify any code, the config updated automatically for sharded training:

```diff
optimizer:
   _target_: torch.optim.AdamW
   lr: ${training.lr}
trainer:
   process_position: 0
   num_nodes: 1
   num_processes: 1
-  gpus: null
+  gpus: 1
   auto_select_gpus: false
   tpu_cores: null
   log_gpu_memory: null
   ...
   val_check_interval: 1.0
   flush_logs_every_n_steps: 100
   log_every_n_steps: 50
-  accelerator: null
+  accelerator: ddp
   sync_batchnorm: false
-  precision: 32
+  precision: 16
   weights_summary: top
   weights_save_path: null
   num_sanity_val_steps: 2
   ....
   terminate_on_nan: false
   auto_scale_batch_size: false
   prepare_data_per_node: true
-  plugins: null
+  plugins:
+    _target_: pytorch_lightning.plugins.DDPShardedPlugin
   amp_backend: native
   amp_level: O2
   move_metrics_to_cpu: false
tokenizer:
   pretrained_model_name_or_path: ${backbone.pretrained_model_name_or_path}
   use_fast: true
 backbone:
   pretrained_model_name_or_path: bert-base-cased

```   
</details>

#### Enable DeepSpeed ZeRO-Offload Training.
```bash
python train.py \
    +task=nlp/text_classification \
    +dataset=nlp/text_classification/emotion \
    trainer=zero_offload
```

#### Train roberta-base backbone, on SWAG dataset multiple choice task.
```bash
python train.py \
    +task=nlp/multiple_choice \
    +dataset=nlp/multiple_choice/swag \
    backbone.pretrained_model_name_or_path=roberta-base
```

#### Inference with pre-trained bert-base-cased on SQuAD using question-answering task with 2 GPUs.
```bash
python train.py \
    +task=nlp/question_answering \
    +dataset=nlp/question_answering/squad \
    trainer.gpus=2 \
    training.do_train=False
```

### Custom tasks and datasets

You can train Lightning transformers tasks on your own data files, and you can even create your own datasets for custim processing and your own tasks. Read more in our docs.

## Contribute

Pull requests are welcome. For major changes, please open an issue first to discuss what you would like to change.

Please make sure to update tests as appropriate.

## Community
For help or questions, join our huge community on [Slack](https://join.slack.com/t/pytorch-lightning/shared_invite/zt-f6bl2l0l-JYMK3tbAgAmGRrlNr00f1A)!

## License
Please observe the Apache 2.0 license that is listed in this repository. In addition, the Lightning framework is Patent Pending.
