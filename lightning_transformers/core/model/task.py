from typing import Dict, Any, Optional

import torch

from lightning_transformers.core.model import LitTransformer


class TaskTransformer(LitTransformer):
    """
    Base class for task specific transformers
    """

    def __init__(
        self,
        model: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        scheduler: torch.optim.lr_scheduler._LRScheduler,
        tokenizer: Optional[Any] = None,
    ):
        self.tokenizer = tokenizer
        super().__init__(model, optimizer, scheduler)

    def setup(self, stage):
        self.configure_metrics()

    def configure_metrics(self):
        """
        Override to configure metrics for train/validation/test.
        This is called on fit start to have access to the data module,
        and initialize any data specific metrics.
        """
        pass

    def on_save_checkpoint(self, checkpoint: Dict[str, Any]) -> Any:
        # Save tokenizer for predictions
        checkpoint["tokenizer"] = self.tokenizer

    def on_load_checkpoint(self, checkpoint: Dict[str, Any]) -> None:
        self.tokenizer = checkpoint["tokenizer"]
