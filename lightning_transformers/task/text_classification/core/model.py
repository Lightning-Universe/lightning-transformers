from typing import Optional

import pytorch_lightning as pl
import torch
from omegaconf import DictConfig

from lightning_transformers.core.model import LitTransformer


class LitTextClassificationTransformer(LitTransformer):
    def __init__(
            self,
            pretrained_model_name_or_path: str,
            model_type: str,
            optim: DictConfig,
            num_classes: int,
            scheduler: Optional[DictConfig] = None):
        super().__init__(
            pretrained_model_name_or_path=pretrained_model_name_or_path,
            model_type=model_type,
            optim=optim,
            scheduler=scheduler
        )
        self._initialize_metrics(num_classes=num_classes)

    def training_step(self, batch, batch_idx):
        del batch['idx']  # Can we hide this? this is given from the HF Feature object
        outputs = self(**batch)
        loss = outputs[0]
        self.log('train_loss', loss)
        return loss

    def validation_step(self, batch, batch_idx, dataloader_idx=0):
        del batch['idx']  # Can we hide this? this is given from the HF Feature object
        outputs = self(**batch)
        val_loss, logits = outputs[:2]
        preds = torch.argmax(logits, axis=1)
        metric_dict = self.calculate_metrics(preds, batch['labels'])
        self.log_dict(metric_dict, prog_bar=True, on_step=False, on_epoch=True)
        self.log('val_loss', val_loss, prog_bar=True)

    def test_step(self, batch, batch_idx, dataloader_idx=0):
        del batch['labels']  #
        idxs = batch.pop('idx')
        outputs = self(**batch)
        logits = outputs[0]
        preds = torch.argmax(logits, axis=1)

        # Should be an option, don't hardcode things in the code.
        self.write_prediction('idxs', idxs, self.hparams.predictions_file)
        self.write_prediction('preds', preds, self.hparams.predictions_file)

    def log_metrics(self, preds, labels, mode='val'):
        p = self.precision_metric(preds, labels)
        r = self.recall_metric(preds, labels)
        a = self.accuracy_metric(preds, labels)
        return {f'{mode}_precision': p, f'{mode}_recall': r, f'{mode}_acc': a}

    def _initialize_metrics(self, num_classes: int):
        self.precision_metric = pl.metrics.Precision(num_classes=num_classes)
        self.recall_metric = pl.metrics.Recall(num_classes=num_classes)
        self.accuracy_metric = pl.metrics.Accuracy()
