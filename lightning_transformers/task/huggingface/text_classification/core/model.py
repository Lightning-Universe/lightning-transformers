from typing import Optional

import pytorch_lightning as pl
import torch
from omegaconf import DictConfig
from transformers import AutoConfig

from lightning_transformers.core.model import LitAutoModelTransformer


class LitAutoModelTextClassificationTransformer(LitAutoModelTransformer):
    def __init__(self,
                 downstream_model_type: str,
                 backbone: DictConfig,
                 optim: DictConfig,
                 num_classes: int,
                 scheduler: Optional[DictConfig] = None):
        self.num_classes = num_classes
        super().__init__(
            downstream_model_type=downstream_model_type,
            backbone=backbone,
            optim=optim,
            scheduler=scheduler
        )
        self._initialize_metrics(num_classes=num_classes)

    def generate_config(self):
        self.config = AutoConfig.from_pretrained(
            self.hparams.backbone.pretrained_model_name_or_path,
            num_labels=self.num_classes
        )

    def training_step(self, batch, batch_idx):
        del batch['idx']  # Can we hide this? this is given from the HF Feature object
        outputs = self(**batch)
        loss = outputs[0]
        self.log('train_loss', loss)
        return loss

    def validation_step(self, batch, batch_idx, dataloader_idx=0):
        batch = self._remove_unneeded_batch_keys(batch)
        outputs = self(**batch)
        val_loss, logits = outputs[:2]
        preds = torch.argmax(logits, axis=1)
        metric_dict = self._calculate_metrics(preds, batch['labels'])
        self.log_dict(metric_dict, prog_bar=True, on_step=False, on_epoch=True)
        self.log('val_loss', val_loss, prog_bar=True, sync_dist=True)

    def test_step(self, batch, batch_idx, dataloader_idx=0):
        batch = self._remove_unneeded_batch_keys(batch)
        outputs = self(**batch)
        val_loss, logits = outputs[:2]
        preds = torch.argmax(logits, axis=1)
        metric_dict = self._calculate_metrics(preds, batch['labels'], mode='test')
        self.log_dict(metric_dict, prog_bar=True, on_step=False, on_epoch=True)
        self.log('test_loss', val_loss, prog_bar=True, sync_dist=True)

    def _remove_unneeded_batch_keys(self, batch):
        del batch['idx']
        return batch

    def _initialize_metrics(self, num_classes: int):
        self.precision_metric = pl.metrics.Precision(num_classes=num_classes)
        self.recall_metric = pl.metrics.Recall(num_classes=num_classes)
        self.accuracy_metric = pl.metrics.Accuracy()

    def _calculate_metrics(self, preds, labels, mode='val'):
        # Not required by all models. Only required for classification
        p = self.precision_metric(preds, labels)
        r = self.recall_metric(preds, labels)
        a = self.accuracy_metric(preds, labels)
        return {f'{mode}_precision': p, f'{mode}_recall': r, f'{mode}_acc': a}
