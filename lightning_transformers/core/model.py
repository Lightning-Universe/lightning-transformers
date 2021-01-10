from typing import Dict, Optional, Any

import pytorch_lightning as pl
import torch


class LitTransformer(pl.LightningModule):
    """
    Base class for transformers.
    Provides a few helper functions primarily for optimization.
    """

    def __init__(
        self,
        model: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        scheduler: torch.optim.lr_scheduler._LRScheduler,
    ):
        super().__init__()
        self.model = model
        self.optimizer = optimizer
        self.scheduler = scheduler

    def configure_optimizers(self) -> Dict:
        """Prepare optimizer and scheduler"""
        return {
            "optimizer": self.optimizer,
            "lr_scheduler": {"scheduler": self.scheduler, "interval": "step", "frequency": 1},
        }

    @property
    def num_training_steps(self) -> int:
        """Total training steps inferred from datamodule and devices."""
        if self.trainer.max_steps:
            return self.trainer.max_steps

        dataset_size = (
            self.trainer.limit_train_batches
            if self.trainer.limit_train_batches != 0
            else len(self.trainer.datamodule.train_dataloader())
        )

        num_devices = max(1, self.trainer.num_gpus, self.trainer.num_processes)
        if self.trainer.tpu_cores:
            num_devices = max(num_devices, self.trainer.tpu_cores)

        effective_batch_size = self.trainer.datamodule.batch_size * self.trainer.accumulate_grad_batches * num_devices
        return (dataset_size // effective_batch_size) * self.trainer.max_epochs


class TaskTransformer(LitTransformer):
    """
    Base class for task specific transformers
    """

    def setup(self, stage):
        self.configure_metrics()

    def configure_metrics(self) -> Optional[Any]:
        """
        Override to configure metrics for train/validation/test.
        This is called on fit start to have access to the data module,
        and initialize any data specific metrics.
        """
        pass
