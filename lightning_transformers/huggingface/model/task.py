from dataclasses import dataclass
from typing import Optional

from transformers import PreTrainedTokenizerBase

from lightning_transformers.core.model import TaskTransformer
from lightning_transformers.huggingface.instantiator import Instantiator
from lightning_transformers.huggingface.model.base import TransformerConfig


@dataclass
class TaskTransformerConfig(TransformerConfig):
    tokenizer: Optional[PreTrainedTokenizerBase] = None


class HFTaskTransformer(TaskTransformer):
    """
    Base class for task specific transformers, wrapping pre-trained language models for downstream tasks.
    The API is built on top of AutoModel and AutoConfig, provided by HuggingFace.

    see: https://huggingface.co/transformers/model_doc/auto.html
    """

    def __init__(self, instantiator: Instantiator, config: TaskTransformerConfig):
        model = instantiator.model(config.model)
        optimizer = instantiator.optimizer(model, config.optimizer)
        scheduler = instantiator.scheduler(config.scheduler, optimizer)
        super().__init__(model, optimizer, scheduler, tokenizer=config.tokenizer)
