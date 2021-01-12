from dataclasses import dataclass


@dataclass
class TransformerDataConfig:
    batch_size: int
    num_workers: int
