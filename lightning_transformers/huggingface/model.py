from dataclasses import dataclass
from typing import Optional

from omegaconf import DictConfig

from lightning_transformers.core.model import TaskTransformer
from lightning_transformers.huggingface.instantiator import Instantiator


@dataclass
class HFTransformerConfig:
    model: DictConfig
    optimizer: DictConfig
    scheduler: DictConfig
    tokenizer: Optional[DictConfig] = None


class HFTransformer(TaskTransformer):
    """
    Base class for task specific transformers, wrapping pre-trained language models for downstream tasks.
    The API is built on top of AutoModel and AutoConfig, provided by HuggingFace.

    see: https://huggingface.co/transformers/model_doc/auto.html
    """

    def __init__(self, instantiator: Instantiator, cfg: HFTransformerConfig):
        model = instantiator.model(cfg.model)
        optimizer = instantiator.optimizer(model, cfg.optimizer)
        scheduler = instantiator.scheduler(cfg.scheduler, optimizer)
        tokenizer = instantiator.tokenizer(cfg.tokenizer) if cfg.tokenizer is not None else None
        super().__init__(model, optimizer, scheduler, tokenizer=tokenizer)
