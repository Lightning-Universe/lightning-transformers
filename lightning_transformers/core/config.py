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


# dynamically create a config class for the trainer using its signature
def __get_trainer_fields() -> List[Tuple[str, Type, Any]]:
    sig = inspect.signature(pl.Trainer.__init__)
    parameters = list(sig.parameters.values())
    parameters = parameters[1:]  # exclude self
    return [(p.name, p.annotation, p.default) for p in parameters]


@dataclass
class TrainerConfig(make_dataclass("", __get_trainer_fields())):
    __doc__ = pl.Trainer.__init__.__doc__
