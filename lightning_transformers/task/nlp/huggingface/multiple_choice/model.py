import pytorch_lightning as pl
import torch
from torchmetrics import Accuracy, Precision, Recall

from lightning_transformers.core.nlp.huggingface import HFTransformer


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
        metric_dict = self.compute_metrics(preds, batch["labels"], mode=mode)
        self.log_dict(metric_dict, prog_bar=True, on_step=False, on_epoch=True)
        self.log(f"{mode}_loss", loss, prog_bar=True, sync_dist=True)
        return loss

    def configure_metrics(self, stage: str):
        self.prec = Precision(num_classes=self.num_classes)
        self.recall = Recall(num_classes=self.num_classes)
        self.acc = Accuracy()
        self.metrics = {"precision": self.prec, "recall": self.recall, "accuracy": self.acc}

    @property
    def num_classes(self):
        return self.trainer.datamodule.num_classes

    def compute_metrics(self, preds, labels, mode="val"):
        # Remove ignored index (special tokens)
        # Not required by all models. Only required for classification
        return {f"{mode}_{k}": metric(preds, labels) for k, metric in self.metrics.items()}
