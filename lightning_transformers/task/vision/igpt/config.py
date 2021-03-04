from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Union

from lightning_transformers.core.config import TransformerDataConfig


@dataclass
class ImageGPTDataConfig(TransformerDataConfig):
    dataset: str = "cifar10"
    data_dir: Optional[Union[str, Path]] = None
