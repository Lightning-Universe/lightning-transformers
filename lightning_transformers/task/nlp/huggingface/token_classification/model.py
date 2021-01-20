from typing import Any, Dict, List, Optional

import pytorch_lightning as pl
import torch

from lightning_transformers.core.nlp.huggingface import HFTransformer


class TokenClassificationTransformer(HFTransformer):
    def __init__(self, *args, labels: List[str], **kwargs):
        super().__init__(*args, **kwargs, num_labels=len(labels))
        self.labels = labels
        self.metrics = {}

    def training_step(self, batch: Any, batch_idx: int) -> torch.Tensor:
        outputs = self.model(**batch)
        loss = outputs[0]
        self.log("train_loss", loss)
        return loss

    def common_step(self, prefix: str, batch: Any) -> torch.Tensor:
        outputs = self.model(**batch)
        loss, logits = outputs[:2]
        preds = torch.argmax(logits, dim=2)
        metric_dict = self.compute_metrics(preds, batch["labels"], mode=prefix)
        self.log_dict(metric_dict, prog_bar=True, on_step=False, on_epoch=True)
        self.log(f"{prefix}_loss", loss, prog_bar=True, sync_dist=True)
        return loss

    def validation_step(self, batch: Any, batch_idx: int, dataloader_idx: int = 0) -> torch.Tensor:
        return self.common_step("val", batch)

    def test_step(self, batch: Any, batch_idx: int, dataloader_idx: int = 0) -> torch.Tensor:
        return self.common_step("test", batch)

    def configure_metrics(self, _) -> None:
        self.prec = pl.metrics.Precision(num_classes=self.num_labels)
        self.recall = pl.metrics.Recall(num_classes=self.num_labels)
        self.f1 = pl.metrics.F1(num_classes=self.num_labels)
        self.acc = pl.metrics.Accuracy()
        self.metrics = {"precision": self.prec, "recall": self.recall, "accuracy": self.acc, "f1": self.f1}

    @property
    def num_labels(self) -> int:
        return len(self.labels)

    def compute_metrics(self, predictions, labels, mode="val") -> Dict[str, torch.Tensor]:
        # Remove ignored index (special tokens)
        predictions = predictions[labels != -100]
        labels = labels[labels != -100]
        # Not required by all models. Only required for classification
        return {f"{mode}_{k}": metric(predictions, labels) for k, metric in self.metrics.items()}

    @property
    def pipeline_task(self) -> Optional[str]:
        return "ner"
