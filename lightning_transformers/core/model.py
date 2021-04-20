# Copyright The PyTorch Lightning team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
from typing import Any, Dict, Optional, Tuple, Union

import pytorch_lightning as pl
import torch
from pytorch_lightning.utilities import rank_zero_info
from pytorch_lightning.utilities.exceptions import MisconfigurationException

from lightning_transformers.core.config import OptimizerConfig, SchedulerConfig
from lightning_transformers.core.instantiator import Instantiator


class LitTransformer(pl.LightningModule):
    """
    Base class for transformers.
    Provides a few helper functions primarily for optimization.
    """

    def __init__(
        self,
        model: torch.nn.Module,
        optimizer: Optional[torch.optim.Optimizer] = None,
        scheduler: Optional[torch.optim.lr_scheduler._LRScheduler] = None,
    ):
        super().__init__()
        self.model = model
        # some optimizers/schedulers need parameters only known dynamically
        # allow users to override the getter to instantiate them lazily
        self.optimizer = optimizer
        self.scheduler = scheduler

    def configure_optimizers(self) -> Dict:
        """Prepare optimizer and scheduler"""
        return {
            "optimizer": self.optimizer,
            "lr_scheduler": {
                "scheduler": self.scheduler,
                "interval": "step",
                "frequency": 1
            },
        }

    @property
    def num_training_steps(self) -> int:
        """Total training steps inferred from datamodule and devices."""
        if isinstance(self.trainer.limit_train_batches, int) and self.trainer.limit_train_batches != 0:
            dataset_size = self.trainer.limit_train_batches
        elif isinstance(self.trainer.limit_train_batches, float):
            # limit_train_batches is a percentage of batches
            dataset_size = len(self.trainer.datamodule.train_dataloader())
            dataset_size = int(dataset_size * self.trainer.limit_train_batches)
        else:
            dataset_size = len(self.trainer.datamodule.train_dataloader())

        num_devices = max(1, self.trainer.num_gpus, self.trainer.num_processes)
        if self.trainer.tpu_cores:
            num_devices = max(num_devices, self.trainer.tpu_cores)

        effective_batch_size = self.trainer.accumulate_grad_batches * num_devices
        max_estimated_steps = (dataset_size // effective_batch_size) * self.trainer.max_epochs

        if self.trainer.max_steps and self.trainer.max_steps < max_estimated_steps:
            return self.trainer.max_steps
        return max_estimated_steps

    def compute_warmup(self, num_training_steps: int, num_warmup_steps: Union[int, float]) -> Tuple[int, int]:
        if num_training_steps < 0:
            # less than 0 specifies to infer number of training steps
            num_training_steps = self.num_training_steps
        if isinstance(num_warmup_steps, float):
            # Convert float values to percentage of training steps to use as warmup
            num_warmup_steps *= num_training_steps
        return num_training_steps, num_warmup_steps

    def setup(self, stage: str):
        self.configure_metrics(stage)

    def configure_metrics(self, stage: str) -> Optional[Any]:
        """
        Override to configure metrics for train/validation/test.
        This is called on fit start to have access to the data module,
        and initialize any data specific metrics.
        """
        pass


class TaskTransformer(LitTransformer):
    """
    Base class for task specific transformers
    """

    def __init__(
        self,
        model: torch.nn.Module,
        optimizer: OptimizerConfig,
        scheduler: SchedulerConfig,
        instantiator: Optional[Instantiator] = None,
    ):
        super().__init__(model)
        self.instantiator = instantiator
        self.optimizer_cfg = optimizer
        self.scheduler_cfg = scheduler

    def configure_optimizers(self) -> Dict:
        if self.instantiator is None:
            raise MisconfigurationException(
                "To train you must provide an instantiator to instantiate the optimizer and scheduler "
                "or override `configure_optimizers` in the `LightningModule`."
            )
        self.optimizer = self.instantiator.optimizer(self.model, self.optimizer_cfg)
        # compute_warmup needs the datamodule to be available when `self.num_training_steps`
        # is called that is why this is done here and not in the __init__
        self.scheduler_cfg.num_training_steps, self.scheduler_cfg.num_warmup_steps = self.compute_warmup(
            num_training_steps=self.scheduler_cfg.num_training_steps,
            num_warmup_steps=self.scheduler_cfg.num_warmup_steps,
        )
        rank_zero_info(f"Inferring number of training steps, set to {self.scheduler_cfg.num_training_steps}")
        rank_zero_info(f"Inferring number of warmup steps from ratio, set to {self.scheduler_cfg.num_warmup_steps}")
        self.scheduler = self.instantiator.scheduler(self.scheduler_cfg, self.optimizer)
        return super().configure_optimizers()

    def on_save_checkpoint(self, checkpoint: Dict[str, Any]):
        # Save tokenizer from datamodule for predictions
        if self.instantiator:
            checkpoint["instantiator"] = self.instantiator

    def on_load_checkpoint(self, checkpoint: Dict[str, Any]) -> None:
        self.instantiator = checkpoint.get("instantiator")
