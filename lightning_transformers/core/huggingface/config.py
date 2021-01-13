from dataclasses import dataclass
from typing import Optional

from lightning_transformers.core.data import TransformerDataConfig


@dataclass
class HFTransformerDataConfig(TransformerDataConfig):
    dataset_name: Optional[str] = None
    train_val_split: Optional[int] = None
    train_file: Optional[str] = None
    validation_file: Optional[str] = None
    padding: str = "max_length"
    truncation: str = "only_first"
    max_length: int = 128
    preprocessing_num_workers: int = 8
    load_from_cache_file: bool = True
    dataset_config_name: Optional[str] = None


@dataclass
class HFBackboneConfig:
    downstream_model_type: str
    pretrained_model_name_or_path: str


@dataclass
class HFSchedulerConfig:
    num_training_steps: int = -1
    num_warmup_steps: float = 0.1
