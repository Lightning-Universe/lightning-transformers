from typing import Optional

import pytorch_lightning as pl
import torch
from omegaconf import DictConfig

from lightning_transformers.core.model import TaskTransformer


class TextClassificationTransformer(TaskTransformer):
    def __init__(self,
                 downstream_model_type: str,
                 backbone: DictConfig,
                 optim: DictConfig,
                 scheduler: Optional[DictConfig] = None,
                 config_data_args: Optional[dict] = None):
        super().__init__(
            downstream_model_type=downstream_model_type,
            backbone=backbone,
            optim=optim,
            scheduler=scheduler,
            config_data_args=config_data_args
        )

    def training_step(self, batch, batch_idx):
        outputs = self.model(**batch)
        loss = outputs[0]
        self.log('train_loss', loss)
        return loss

    def validation_step(self, batch, batch_idx, dataloader_idx=0):
        outputs = self.model(**batch)
        val_loss, logits = outputs[:2]
        preds = torch.argmax(logits, axis=1)
        metric_dict = self._calculate_metrics(preds, batch['labels'])
        self.log_dict(metric_dict, prog_bar=True, on_step=False, on_epoch=True)
        self.log('val_loss', val_loss, prog_bar=True, sync_dist=True)

    def test_step(self, batch, batch_idx, dataloader_idx=0):
        outputs = self.model(**batch)
        val_loss, logits = outputs[:2]
        preds = torch.argmax(logits, axis=1)
        metric_dict = self._calculate_metrics(preds, batch['labels'], mode='test')
        self.log_dict(metric_dict, prog_bar=True, on_step=False, on_epoch=True)
        self.log('test_loss', val_loss, prog_bar=True, sync_dist=True)

    def configure_metrics(self):
        self.precision_metric = pl.metrics.Precision(num_classes=self.num_classes)
        self.recall_metric = pl.metrics.Recall(num_classes=self.num_classes)
        self.accuracy_metric = pl.metrics.Accuracy()

    @property
    def num_classes(self):
        return self.trainer.datamodule.num_classes

    def _calculate_metrics(self, preds, labels, mode='val'):
        # Not required by all models. Only required for classification
        p = self.precision_metric(preds, labels)
        r = self.recall_metric(preds, labels)
        a = self.accuracy_metric(preds, labels)
        return {f'{mode}_precision': p, f'{mode}_recall': r, f'{mode}_acc': a}
