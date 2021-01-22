from dataclasses import dataclass


@dataclass
class TransformerDataConfig:
    batch_size: int
    num_workers: int


@dataclass
class OptimizerConfig:
    lr: float = 1e-3
    weight_decay: float = 0.0


@dataclass
class SchedulerConfig:
    num_training_steps: int = -1
    num_warmup_steps: float = 0.1
