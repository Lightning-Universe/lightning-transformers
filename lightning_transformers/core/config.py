from dataclasses import dataclass


@dataclass
class TransformerDataConfig:
    batch_size: int
    num_workers: int


@dataclass
class OptimizerConfig:
    lr: float = 1e-3
    weight_decay: float = 0.0
