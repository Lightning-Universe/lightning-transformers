from dataclasses import dataclass
from typing import Optional

from transformers import PreTrainedTokenizerBase

from lightning_transformers.core.model import TaskTransformer
from lightning_transformers.huggingface.model.base import HydraMixin, TransformerConfig


@dataclass
class TaskTransformerConfig(TransformerConfig):
    tokenizer: Optional[PreTrainedTokenizerBase] = None


class HFTaskTransformer(HydraMixin, TaskTransformer):
    """
    Base class for task specific transformers, wrapping pre-trained language models for downstream tasks.
    The API is built on top of AutoModel and AutoConfig, provided by HuggingFace.

    see: https://huggingface.co/transformers/model_doc/auto.html
    """

    def __init__(self, config: TaskTransformerConfig):
        model = self.prepare_model(config.model)
        optimizer = self.prepare_optimzer(model, config.optimizer)
        scheduler = self.prepare_scheduler(config.scheduler, optimizer)
        super().__init__(
            model,
            optimizer,
            scheduler,
            tokenizer=config.tokenizer,
        )
