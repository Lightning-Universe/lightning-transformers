import pytorch_lightning as pl
import torch

from lightning_transformers.core.huggingface import HFTransformer


class MultipleChoiceTransformer(HFTransformer):

    def training_step(self, batch, batch_idx):
        loss = self._step(batch, batch_idx, "train")
        return loss

    def validation_step(self, batch, batch_idx, dataloader_idx=0):
        loss = self._step(batch, batch_idx, "val")
        return loss

    def test_step(self, batch, batch_idx, dataloader_idx=0):
        loss = self._step(batch, batch_idx, "test")
        return loss

    def _step(self, batch, batch_idx, mode):
        outputs = self.model(**batch)
        loss, logits = outputs[:2]
        preds = torch.argmax(logits, axis=1)
        metric_dict = self._calculate_metrics(preds, batch['labels'], mode=mode)
        self.log_dict(metric_dict, prog_bar=True, on_step=False, on_epoch=True)
        self.log(f'{mode}_loss', loss, prog_bar=True, sync_dist=True)
        return loss

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
