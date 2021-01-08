import math
from typing import Optional, Any, Dict, Union

import pytorch_lightning as pl
from pytorch_lightning import _logger as log
from hydra.utils import get_class, instantiate
from omegaconf import DictConfig
from tokenizers import Tokenizer
from transformers import PreTrainedTokenizerFast, PreTrainedTokenizer


class LitTransformer(pl.LightningModule):
    """
    Base class for transformers.
    Provides a few helper functions primarily for optimization.
    """

    def __init__(self,
                 model: Any,
                 optim: DictConfig,
                 scheduler: DictConfig):
        super().__init__()
        self.model = model
        self.scheduler = scheduler
        self.optim = optim

    def configure_optimizers(self):
        """Prepare optimizer and schedule (linear warmup and decay)"""
        no_decay = ["bias", "LayerNorm.weight"]
        optimizer_grouped_parameters = [
            {
                "params": [p for n, p in self.model.named_parameters() if not any(nd in n for nd in no_decay)],
                "weight_decay": self.hparams.optim.weight_decay,
            },
            {
                "params": [p for n, p in self.model.named_parameters() if any(nd in n for nd in no_decay)],
                "weight_decay": 0.0,
            },
        ]
        optimizer = instantiate(self.optim, optimizer_grouped_parameters)

        if self.scheduler.num_training_steps < 0:
            # less than 0 specifies to infer number of training steps
            self.scheduler.num_training_steps = self.num_training_steps
            log.info(f"Inferring number of training steps, set to {self.scheduler.num_training_steps}")

        if isinstance(self.scheduler.num_warmup_steps, float):
            # Convert float values to percentage of training steps to use as warmup
            warmup_ratio = self.scheduler.num_warmup_steps
            self.scheduler.num_warmup_steps = self.scheduler.num_training_steps * warmup_ratio
            log.info(f"Inferring number of warmup steps from ratio, set to {self.scheduler.num_warmup_steps}")

        scheduler = instantiate(
            config=self.scheduler,
            optimizer=optimizer
        )
        scheduler = {'scheduler': scheduler, 'interval': 'step', 'frequency': 1}
        return [optimizer], [scheduler]

    @property
    def num_training_steps(self) -> int:
        """Total training steps inferred from datamodule and devices."""
        if self.trainer.max_steps:
            return self.trainer.max_steps
        batch_size = self.trainer.datamodule.batch_size

        if self.trainer.limit_train_batches != 0:
            dataset_size = self.trainer.limit_train_batches
        else:
            dataset_size = len(self.trainer.datamodule.train_dataloader())

        num_devices = max(1, self.trainer.num_gpus, self.trainer.num_processes)
        if self.trainer.tpu_cores:
            num_devices = max(num_devices, self.trainer.tpu_cores)

        effective_batch_size = batch_size * self.trainer.accumulate_grad_batches * num_devices
        return math.ceil(dataset_size / effective_batch_size) * self.trainer.max_epochs


class TaskTransformer(LitTransformer):
    """
    Base class for task specific transformers, wrapping pre-trained language models for downstream tasks.
    The API is built on top of AutoModel and AutoConfig, provided by HuggingFace.

    see: https://huggingface.co/transformers/model_doc/auto.html
    """

    def __init__(
            self,
            downstream_model_type: str,
            backbone: DictConfig,
            optim: DictConfig,
            tokenizer: Union[Tokenizer, PreTrainedTokenizer, PreTrainedTokenizerFast],
            scheduler: Optional[DictConfig] = None,
            config_data_args: Optional[dict] = None):
        # Resolve the bug in Lightning save_hyperparameters
        optim.lr = optim.lr  # todo wat

        self.tokenizer = tokenizer  # tokenizer saving handled via on save/load hooks
        self.save_hyperparameters(
            'downstream_model_type',
            'backbone',
            'optim',
            'scheduler',
            'config_data_args'
        )

        model = get_class(self.hparams.downstream_model_type).from_pretrained(
            self.hparams.backbone.pretrained_model_name_or_path,
            **self.hparams.config_data_args
        )
        super().__init__(
            model=model,
            optim=self.hparams.optim,
            scheduler=self.hparams.scheduler
        )

    def setup(self, stage):
        self.configure_metrics()

    def configure_metrics(self):
        """
        Override to configure metrics for train/validation/test.
        This is called on fit start to have access to the data module,
        and initialize any data specific metrics.
        """
        pass

    def on_save_checkpoint(self, checkpoint: Dict[str, Any]):
        # Save tokenizer from datamodule for predictions
        checkpoint['tokenizer'] = self.tokenizer

    def on_load_checkpoint(self, checkpoint: Dict[str, Any]) -> None:
        self.tokenizer = checkpoint['tokenizer']
