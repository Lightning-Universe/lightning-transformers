import torch
from typing import Optional
from omegaconf import DictConfig
import pytorch_lightning as pl
from lightning_transformers.core.model import LitTransformer


class LitMultipleChoiceTransformer(LitTransformer):
    def __init__(
            self,
            downstream_model_type: str,
            model: DictConfig,
            optim: DictConfig,
            num_classes: int,
            scheduler: Optional[DictConfig] = None):
        super().__init__(
            downstream_model_type=downstream_model_type,
            model=model,
            optim=optim,
            scheduler=scheduler
        )
        self._initialize_metrics(num_classes=num_classes)

    def _step(self, batch, batch_idx, mode):
        batch = self._remove_unneeded_batch_keys(batch)
        outputs = self(**batch)
        loss, logits = outputs[:2]
        preds = torch.argmax(logits, axis=1)
        metric_dict = self._calculate_metrics(preds, batch['labels'], mode=mode)
        self.log_dict(metric_dict, prog_bar=True, on_step=False, on_epoch=True)
        self.log(f'{mode}_loss', loss, prog_bar=True, sync_dist=True)  
        return loss

    def training_step(self, batch, batch_idx):
        loss = self._step(batch, batch_idx, "train")
        return loss

    def validation_step(self, batch, batch_idx, dataloader_idx=0):
        loss = self._step(batch, batch_idx, "val")
        return loss

    def test_step(self, batch, batch_idx, dataloader_idx=0):
        loss = self._step(batch, batch_idx, "test")
        return loss

    def _remove_unneeded_batch_keys(self, batch):
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