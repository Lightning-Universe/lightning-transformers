# Copyright The PyTorch Lightning team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Union

from datasets import Version


@dataclass
class TransformerDataConfig:
    batch_size: int = 32
    num_workers: int = 0
    dataset_name: Optional[str] = None
    dataset_config_name: Optional[str] = None
    revision: Optional[Union[str, Version]] = None
    train_val_split: Optional[int] = None
    train_file: Optional[str] = None
    test_file: Optional[str] = None
    validation_file: Optional[str] = None
    padding: Union[str, bool] = "max_length"
    truncation: str = "only_first"
    max_length: int = 128
    preprocessing_num_workers: int = 1
    load_from_cache_file: bool = True
    cache_dir: Optional[Union[Path, str]] = None
    limit_train_samples: Optional[int] = None
    limit_val_samples: Optional[int] = None
    limit_test_samples: Optional[int] = None
    train_subset_name: Optional[str] = None
    validation_subset_name: Optional[str] = None
    test_subset_name: Optional[str] = None
    streaming: bool = False


@dataclass
class OptimizerConfig:
    lr: float = 1e-5
    weight_decay: float = 0.0


@dataclass
class SchedulerConfig:
    num_training_steps: int = -1
    num_warmup_steps: float = 0.1


@dataclass
class TrainerConfig:
    ...


@dataclass
class BackboneConfig:
    pretrained_model_name_or_path: Optional[str] = None


@dataclass
class TaskConfig:
    optimizer: OptimizerConfig = OptimizerConfig()
    scheduler: SchedulerConfig = SchedulerConfig()
    downstream_model_type: Optional[str] = None
    backbone: BackboneConfig = BackboneConfig()


@dataclass
class TokenizerConfig:
    downstream_model_type: Optional[str] = None
    pretrained_model_name_or_path: Optional[str] = None
    use_fast: bool = True
