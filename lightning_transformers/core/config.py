import inspect
from dataclasses import dataclass, make_dataclass
from typing import Any, List, Optional, Tuple, Type

import pytorch_lightning as pl


@dataclass
class HydraConfig:
    _target_: Optional[str] = None


@dataclass
class TransformerDataConfig(HydraConfig):
    batch_size: int = 32
    num_workers: int = 0


@dataclass
class OptimizerConfig(HydraConfig):
    lr: float = 1e-3
    weight_decay: float = 0.0


@dataclass
class SchedulerConfig(HydraConfig):
    ...


@dataclass
class TrainerConfig(HydraConfig):
    ...
