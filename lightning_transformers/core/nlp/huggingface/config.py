from dataclasses import dataclass
from typing import List, Optional, Union

from lightning_transformers.core.config import OptimizerConfig, SchedulerConfig, TaskConfig
from lightning_transformers.core.data import TransformerDataConfig
from lightning_transformers.core.nlp.config import TokenizerConfig


@dataclass
class HFTransformerDataConfig(TransformerDataConfig):
    dataset_name: Optional[str] = None
    dataset_config_name: Optional[str] = None
    train_val_split: Optional[int] = None
    split: Optional[Union[List, str]] = None
    train_file: Optional[str] = None
    validation_file: Optional[str] = None
    padding: Union[str, bool] = "max_length"
    truncation: str = "only_first"
    max_length: int = 128
    preprocessing_num_workers: int = 8
    load_from_cache_file: bool = True
    cache_dir: Optional[str] = None


@dataclass
class HFTokenizerConfig(TokenizerConfig):
    downstream_model_type: Optional[str] = None
    pretrained_model_name_or_path: Optional[str] = None
    use_fast: bool = True


@dataclass
class HFBackboneConfig:
    downstream_model_type: Optional[str] = None
    pretrained_model_name_or_path: Optional[str] = None


@dataclass
class HFTaskConfig(TaskConfig):
    backbone: HFBackboneConfig = HFBackboneConfig()
    optimizer: OptimizerConfig = OptimizerConfig()
    scheduler: SchedulerConfig = SchedulerConfig()
