from typing import Any, Dict

import pytorch_lightning as pl
import torch

from lightning_transformers.core.nlp.huggingface import HFTransformer


class TextClassificationTransformer(HFTransformer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.metrics = {}

    def training_step(self, batch: Any, batch_idx: int) -> torch.Tensor:
        outputs = self.model(**batch)
        loss = outputs[0]
        self.log("train_loss", loss)
        return loss

    def common_step(self, prefix: str, batch: Any) -> torch.Tensor:
        outputs = self.model(**batch)
        loss, logits = outputs[:2]
        preds = torch.argmax(logits, dim=1)
        metric_dict = self.compute_metrics(preds, batch["labels"], mode=prefix)
        self.log_dict(metric_dict, prog_bar=True, on_step=False, on_epoch=True)
        self.log(f"{prefix}_loss", loss, prog_bar=True, sync_dist=True)
        return loss

    def validation_step(self, batch: Any, batch_idx: int, dataloader_idx: int = 0) -> torch.Tensor:
        return self.common_step("val", batch)

    def test_step(self, batch: Any, batch_idx: int, dataloader_idx: int = 0) -> torch.Tensor:
        return self.common_step("test", batch)

    def configure_metrics(self, _) -> None:
        self.prec = pl.metrics.Precision(num_classes=self.num_classes)
        self.recall = pl.metrics.Recall(num_classes=self.num_classes)
        self.acc = pl.metrics.Accuracy()
        self.metrics = {"precision": self.prec, "recall": self.recall, "accuracy": self.acc}

    @property
    def num_classes(self) -> int:
        return self.trainer.datamodule.num_classes

    def compute_metrics(self, preds, labels, mode="val") -> Dict[str, torch.Tensor]:
        # Not required by all models. Only required for classification
        return {f"{mode}_{k}": metric(preds, labels) for k, metric in self.metrics.items()}
