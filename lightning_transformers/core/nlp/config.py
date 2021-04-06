from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Union

from lightning_transformers.core.config import TaskConfig
from lightning_transformers.core.data import TransformerDataConfig


@dataclass
class HFTransformerDataConfig(TransformerDataConfig):

    dataset_name: Optional[str] = None
    dataset_config_name: Optional[str] = None
    train_val_split: Optional[int] = None
    train_file: Optional[str] = None
    test_file: Optional[str] = None
    validation_file: Optional[str] = None
    padding: Union[str, bool] = "max_length"
    truncation: str = "only_first"
    max_length: int = 128
    preprocessing_num_workers: int = 8
    load_from_cache_file: bool = True
    cache_dir: Optional[Union[Path, str]] = None
    limit_train_samples: Optional[int] = None
    limit_val_samples: Optional[int] = None
    limit_test_samples: Optional[int] = None


@dataclass
class HFTokenizerConfig:
    downstream_model_type: Optional[str] = None
    pretrained_model_name_or_path: Optional[str] = None
    use_fast: bool = True


@dataclass
class HFBackboneConfig:
    pretrained_model_name_or_path: Optional[str] = None


@dataclass
class HFTaskConfig(TaskConfig):
    downstream_model_type: Optional[str] = None
    backbone: HFBackboneConfig = HFBackboneConfig()
