<div align="center">

<img src="docs/source/_static/images/logo.png" width="500px">

**Flexible interface for high performance research using SOTA Transformers leveraging [Pytorch Lightning](https://github.com/PyTorchLightning/pytorch-lightning), [Transformers](https://github.com/huggingface/transformers), and [Hydra](https://github.com/facebookresearch/hydra).**

<img src="https://pl-bolts-doc-images.s3.us-east-2.amazonaws.com/lit-tfmrs.gif">

______________________________________________________________________

<p align="center">
  <a href="#what-is-lightning-transformers">What is Lightning Transformers</a> •
  <a href="#using-lightning-transformers">Using Lightning Transformers</a> •
  <a href="https://lightning-transformers.readthedocs.io/">Docs</a> •
  <a href="#community">Community</a> •
  <a href="#license">License</a>
</p>

______________________________________________________________________

</div>

## Installation

#### Option 1: from PyPI

```bash
pip install lightning-transformers
# instead of: `python train.py ...`, run with:
pl-transformers-train ...
```

#### Option 2: from source

```bash
git clone https://github.com/PyTorchLightning/lightning-transformers.git
cd lightning-transformers
pip install .
python train.py ...
# the `pl-transformers-train` endpoint is also available!
```

</details>

______________________________________________________________________

## What is Lightning-Transformers

Lightning Transformers offers a flexible interface for training and fine-tuning SOTA Transformer models using the [PyTorch Lightning Trainer](https://pytorch-lightning.readthedocs.io/en/stable/common/trainer.html).

- **Train using [HuggingFace Transformers](https://github.com/huggingface/transformers) models and datasets** with Lightning custom Callbacks, Loggers, Accelerators and high performance scaling.
- **Seamless Memory and Speed Optimizations** such as [DeepSpeed ZeRO](https://pytorch-lightning.readthedocs.io/en/latest/multi_gpu.html#deepspeed) or [FairScale Sharded Training](https://pytorch-lightning.readthedocs.io/en/latest/multi_gpu.html#sharded-training) with no code changes.
- **Powerful config composition backed by [Hydra](https://hydra.cc/)** - Easily swap out models, optimizers, schedulers and many more configurations without touching the code.
- **Transformer Task Abstraction for Rapid Research & Experimentation** - Built from the ground up to be task agnostic, the library supports creating transformer tasks across all modalities with little friction.

Lightning Transformers tasks allow you to train models using HuggingFace Transformer models and datasets, use Hydra to hotswap models, optimizers or schedulers and leverage all the advances features that Lightning has to offer, including custom Callbacks, Loggers, Accelerators and high performance scaling with minimal changes.

## Using Lightning-Transformers

**Grid** is our platform for training models at scale on the cloud! Sign up [here](https://www.grid.ai/).

| Task                                                                                                                | Quick Commands                                                                                                           | Run                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                              |
| ------------------------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------ | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| [Language Modeling](https://lightning-transformers.readthedocs.io/en/latest/tasks/nlp/language_modeling.html)       | `python train.py task=nlp/language_modeling dataset=nlp/language_modeling/wikitext trainer.gpus=1 training.batch_size=8` | [![Grid](https://img.shields.io/badge/rid_AI-run-78FF96.svg?labelColor=black&logo=data:image/svg%2bxml%3Bbase64,PHN2ZyB3aWR0aD0iNDgiIGhlaWdodD0iNDgiIGZpbGw9Im5vbmUiIHhtbG5zPSJodHRwOi8vd3d3LnczLm9yZy8yMDAwL3N2ZyI+PHBhdGggZD0iTTEgMTR2MjBhMTQgMTQgMCAwMDE0IDE0aDlWMzYuOEgxMi42VjExaDIyLjV2N2gxMS4yVjE0QTE0IDE0IDAgMDAzMi40IDBIMTVBMTQgMTQgMCAwMDEgMTR6IiBmaWxsPSIjZmZmIi8+PHBhdGggZD0iTTM1LjIgNDhoMTEuMlYyNS41SDIzLjl2MTEuM2gxMS4zVjQ4eiIgZmlsbD0iI2ZmZiIvPjwvc3ZnPg==)](https://platform.grid.ai/#/runs?script=https://github.com/PyTorchLightning/lightning-transformers/blob/ebef2896cca8380ab69873f6c2ee3a08464d2fe3/train.py&cloud=grid&instance=p3.2xlarge&accelerators=1&disk_size=200&framework=lightning&script_args=train.py+task=nlp/language_modeling+dataset=nlp/language_modeling/wikitext+trainer.gpus=1+training.batch_size=8) |
| [Multiple Choice](https://lightning-transformers.readthedocs.io/en/latest/tasks/nlp/multiple_choice.html)           | `python train.py task=nlp/multiple_choice dataset=nlp/multiple_choice/race trainer.gpus=1`                               | [![Grid](https://img.shields.io/badge/rid_AI-run-78FF96.svg?labelColor=black&logo=data:image/svg%2bxml%3Bbase64,PHN2ZyB3aWR0aD0iNDgiIGhlaWdodD0iNDgiIGZpbGw9Im5vbmUiIHhtbG5zPSJodHRwOi8vd3d3LnczLm9yZy8yMDAwL3N2ZyI+PHBhdGggZD0iTTEgMTR2MjBhMTQgMTQgMCAwMDE0IDE0aDlWMzYuOEgxMi42VjExaDIyLjV2N2gxMS4yVjE0QTE0IDE0IDAgMDAzMi40IDBIMTVBMTQgMTQgMCAwMDEgMTR6IiBmaWxsPSIjZmZmIi8+PHBhdGggZD0iTTM1LjIgNDhoMTEuMlYyNS41SDIzLjl2MTEuM2gxMS4zVjQ4eiIgZmlsbD0iI2ZmZiIvPjwvc3ZnPg==)](https://platform.grid.ai/#/runs?script=https://github.com/PyTorchLightning/lightning-transformers/blob/ebef2896cca8380ab69873f6c2ee3a08464d2fe3/train.py&cloud=grid&instance=p3.2xlarge&accelerators=1&disk_size=200&framework=lightning&script_args=train.py+task=nlp/multiple_choice+dataset=nlp/multiple_choice/race+trainer.gpus=1)                               |
| [Question Answering](https://lightning-transformers.readthedocs.io/en/latest/tasks/nlp/question_answering.html)     | `python train.py task=nlp/question_answering dataset=nlp/question_answering/squad trainer.gpus=1`                        | [![Grid](https://img.shields.io/badge/rid_AI-run-78FF96.svg?labelColor=black&logo=data:image/svg%2bxml%3Bbase64,PHN2ZyB3aWR0aD0iNDgiIGhlaWdodD0iNDgiIGZpbGw9Im5vbmUiIHhtbG5zPSJodHRwOi8vd3d3LnczLm9yZy8yMDAwL3N2ZyI+PHBhdGggZD0iTTEgMTR2MjBhMTQgMTQgMCAwMDE0IDE0aDlWMzYuOEgxMi42VjExaDIyLjV2N2gxMS4yVjE0QTE0IDE0IDAgMDAzMi40IDBIMTVBMTQgMTQgMCAwMDEgMTR6IiBmaWxsPSIjZmZmIi8+PHBhdGggZD0iTTM1LjIgNDhoMTEuMlYyNS41SDIzLjl2MTEuM2gxMS4zVjQ4eiIgZmlsbD0iI2ZmZiIvPjwvc3ZnPg==)](https://platform.grid.ai/#/runs?script=https://github.com/PyTorchLightning/lightning-transformers/blob/ebef2896cca8380ab69873f6c2ee3a08464d2fe3/train.py&cloud=grid&instance=p3.2xlarge&accelerators=1&disk_size=200&framework=lightning&script_args=train.py+task=nlp/question_answering+dataset=nlp/question_answering/squad+trainer.gpus=1)                        |
| [Summarization](https://lightning-transformers.readthedocs.io/en/latest/tasks/nlp/summarization.html)               | `python train.py task=nlp/summarization dataset=nlp/summarization/xsum trainer.gpus=1`                                   | [![Grid](https://img.shields.io/badge/rid_AI-run-78FF96.svg?labelColor=black&logo=data:image/svg%2bxml%3Bbase64,PHN2ZyB3aWR0aD0iNDgiIGhlaWdodD0iNDgiIGZpbGw9Im5vbmUiIHhtbG5zPSJodHRwOi8vd3d3LnczLm9yZy8yMDAwL3N2ZyI+PHBhdGggZD0iTTEgMTR2MjBhMTQgMTQgMCAwMDE0IDE0aDlWMzYuOEgxMi42VjExaDIyLjV2N2gxMS4yVjE0QTE0IDE0IDAgMDAzMi40IDBIMTVBMTQgMTQgMCAwMDEgMTR6IiBmaWxsPSIjZmZmIi8+PHBhdGggZD0iTTM1LjIgNDhoMTEuMlYyNS41SDIzLjl2MTEuM2gxMS4zVjQ4eiIgZmlsbD0iI2ZmZiIvPjwvc3ZnPg==)](https://platform.grid.ai/#/runs?script=https://github.com/PyTorchLightning/lightning-transformers/blob/ebef2896cca8380ab69873f6c2ee3a08464d2fe3/train.py&cloud=grid&instance=p3.2xlarge&accelerators=1&disk_size=200&framework=lightning&script_args=train.py+task=nlp/summarization+dataset=nlp/summarization/xsum+trainer.gpus=1)                                   |
| [Text Classification](https://lightning-transformers.readthedocs.io/en/latest/tasks/nlp/text_classification.html)   | `python train.py task=nlp/text_classification dataset=nlp/text_classification/emotion trainer.gpus=1`                    | [![Grid](https://img.shields.io/badge/rid_AI-run-78FF96.svg?labelColor=black&logo=data:image/svg%2bxml%3Bbase64,PHN2ZyB3aWR0aD0iNDgiIGhlaWdodD0iNDgiIGZpbGw9Im5vbmUiIHhtbG5zPSJodHRwOi8vd3d3LnczLm9yZy8yMDAwL3N2ZyI+PHBhdGggZD0iTTEgMTR2MjBhMTQgMTQgMCAwMDE0IDE0aDlWMzYuOEgxMi42VjExaDIyLjV2N2gxMS4yVjE0QTE0IDE0IDAgMDAzMi40IDBIMTVBMTQgMTQgMCAwMDEgMTR6IiBmaWxsPSIjZmZmIi8+PHBhdGggZD0iTTM1LjIgNDhoMTEuMlYyNS41SDIzLjl2MTEuM2gxMS4zVjQ4eiIgZmlsbD0iI2ZmZiIvPjwvc3ZnPg==)](https://platform.grid.ai/#/runs?script=https://github.com/PyTorchLightning/lightning-transformers/blob/ebef2896cca8380ab69873f6c2ee3a08464d2fe3/train.py&cloud=grid&instance=p3.2xlarge&accelerators=1&disk_size=200&framework=lightning&script_args=train.py+task=nlp/text_classification+dataset=nlp/text_classification/emotion+trainer.gpus=1)                    |
| [Token Classification](https://lightning-transformers.readthedocs.io/en/latest/tasks/nlp/token_classification.html) | `python train.py task=nlp/token_classification dataset=nlp/token_classification/conll trainer.gpus=1`                    | [![Grid](https://img.shields.io/badge/rid_AI-run-78FF96.svg?labelColor=black&logo=data:image/svg%2bxml%3Bbase64,PHN2ZyB3aWR0aD0iNDgiIGhlaWdodD0iNDgiIGZpbGw9Im5vbmUiIHhtbG5zPSJodHRwOi8vd3d3LnczLm9yZy8yMDAwL3N2ZyI+PHBhdGggZD0iTTEgMTR2MjBhMTQgMTQgMCAwMDE0IDE0aDlWMzYuOEgxMi42VjExaDIyLjV2N2gxMS4yVjE0QTE0IDE0IDAgMDAzMi40IDBIMTVBMTQgMTQgMCAwMDEgMTR6IiBmaWxsPSIjZmZmIi8+PHBhdGggZD0iTTM1LjIgNDhoMTEuMlYyNS41SDIzLjl2MTEuM2gxMS4zVjQ4eiIgZmlsbD0iI2ZmZiIvPjwvc3ZnPg==)](https://platform.grid.ai/#/runs?script=https://github.com/PyTorchLightning/lightning-transformers/blob/ebef2896cca8380ab69873f6c2ee3a08464d2fe3/train.py&cloud=grid&instance=p3.2xlarge&accelerators=1&disk_size=200&framework=lightning&script_args=train.py+task=nlp/token_classification+dataset=nlp/token_classification/conll+trainer.gpus=1)                    |
| [Translation](https://lightning-transformers.readthedocs.io/en/latest/tasks/nlp/translation.html)                   | `python train.py task=nlp/translation dataset=nlp/translation/wmt16 trainer.gpus=1`                                      | [![Grid](https://img.shields.io/badge/rid_AI-run-78FF96.svg?labelColor=black&logo=data:image/svg%2bxml%3Bbase64,PHN2ZyB3aWR0aD0iNDgiIGhlaWdodD0iNDgiIGZpbGw9Im5vbmUiIHhtbG5zPSJodHRwOi8vd3d3LnczLm9yZy8yMDAwL3N2ZyI+PHBhdGggZD0iTTEgMTR2MjBhMTQgMTQgMCAwMDE0IDE0aDlWMzYuOEgxMi42VjExaDIyLjV2N2gxMS4yVjE0QTE0IDE0IDAgMDAzMi40IDBIMTVBMTQgMTQgMCAwMDEgMTR6IiBmaWxsPSIjZmZmIi8+PHBhdGggZD0iTTM1LjIgNDhoMTEuMlYyNS41SDIzLjl2MTEuM2gxMS4zVjQ4eiIgZmlsbD0iI2ZmZiIvPjwvc3ZnPg==)](https://platform.grid.ai/#/runs?script=https://github.com/PyTorchLightning/lightning-transformers/blob/ebef2896cca8380ab69873f6c2ee3a08464d2fe3/train.py&cloud=grid&instance=p3.2xlarge&accelerators=1&disk_size=200&framework=lightning&script_args=train.py+task=nlp/translation+dataset=nlp/translation/wmt16+trainer.gpus=1)                                      |
|                                                                                                                     |                                                                                                                          | <img width=200/>                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                 |

### Quick recipes

#### Train [bert-base-cased](https://huggingface.co/bert-base-cased) on the [CARER](https://huggingface.co/datasets/emotion) emotion dataset using the Text Classification task.

```bash
python train.py \
    task=nlp/text_classification \
    dataset=nlp/text_classification/emotion
```

<details>
  <summary>See the composed Hydra config used under-the-hood</summary>

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
  run_test_after_fit: true
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
  resume_from_checkpoint: null
  profiler: null
  benchmark: false
  deterministic: false
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
    task=nlp/text_classification \
    dataset=nlp/text_classification/emotion
    backbone.pretrained_model_name_or_path=roberta-base
    optimizer=rmsprop
```

<details>
  <summary>See the changed Hydra config under-the-hood</summary>

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

#### Enable [Sharded](https://pytorch-lightning.readthedocs.io/en/latest/advanced/multi_gpu.html#sharded-training) Training.

```bash
python train.py \
    task=nlp/text_classification \
    dataset=nlp/text_classification/emotion \
    trainer=sharded
```

<details>
  <summary>See the changed Hydra config under-the-hood</summary>
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
   log_every_n_steps: 50
-  accelerator: null
+  accelerator: ddp
   sync_batchnorm: false
-  precision: 32
+  precision: 16
   weights_summary: top
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

#### Enable [DeepSpeed ZeRO](https://pytorch-lightning.readthedocs.io/en/latest/advanced/multi_gpu.html#deepspeed-zero-stage-2) Training.

```bash
python train.py \
    task=nlp/text_classification \
    dataset=nlp/text_classification/emotion \
    trainer=deepspeed
```

<details>
  <summary>See the changed Hydra config under-the-hood</summary>
Without the need to modify any code, the config updated automatically for DeepSpeed:

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
   ...
-  plugins: null
+  plugins:
+    _target_: pytorch_lightning.plugins.DeepSpeedPlugin
+    stage: 2
+    cpu_offload: true
   amp_backend: native
   amp_level: O2
   move_metrics_to_cpu: false
...
```

</details>

#### Train with a pre-trained [t5-base](https://huggingface.co/t5-base) backbone, on the [XSUM](https://huggingface.co/datasets/xsum) dataset using the Summarization task.

```bash
python train.py \
    task=nlp/summarization \
    dataset=nlp/summarization/xsum \
    backbone.pretrained_model_name_or_path=t5-base
```

#### Train with a pre-trained [mt5-base](https://huggingface.co/google/mt5-base) backbone, on the [WMT16](https://huggingface.co/datasets/wmt16) dataset using the Translation task with 2 GPUs.

```bash
python train.py \
    task=nlp/translation \
    dataset=nlp/translation/wmt16 \
    backbone.pretrained_model_name_or_path=google/mt5-base \
    trainer.gpus=2
```

### Custom Files & Datasets

You can train, validate and test Lightning transformers tasks on your own data files, and you can extend datasets for custom processing and your own tasks.

#### [How to train, validate and test on custom files](https://lightning-transformers.readthedocs.io/en/latest/datasets/custom_data.html)

#### [How to extend datasets](https://lightning-transformers.readthedocs.io/en/latest/advanced/custom_datasets.html)

### Custom Tasks

#### [Extending the Language Modeling Task](https://lightning-transformers.readthedocs.io/en/latest/advanced/custom_task.html)

## Contribute

Pull requests are welcome. For major changes, please open an issue first to discuss what you would like to change.

Please make sure to update tests as appropriate.

## Community

For help or questions, join our huge community on [Slack](https://join.slack.com/t/pytorch-lightning/shared_invite/zt-f6bl2l0l-JYMK3tbAgAmGRrlNr00f1A)!

## License

Please observe the Apache 2.0 license that is listed in this repository. In addition, the Lightning framework is Patent Pending.
