import torch
from typing import Optional
from omegaconf import DictConfig
import pytorch_lightning as pl
from lightning_transformers.core.model import LitAutoModelTransformer


class LitLanguageModelingTransformer(LitAutoModelTransformer):

    def on_fit_start(self):
        tokenizer = self.trainer.datamodule.tokenizer
        tokenizer_length = len(tokenizer)
        self.model.resize_token_embeddings(tokenizer_length)
        #self._initialize_metrics(tokenizer_length)

    def _step(self, batch, batch_idx, mode: str = "train"):
        is_train = mode == "train"
        outputs = self(**batch)
        loss, logits = outputs[:2]
        preds = torch.argmax(logits, axis=-1)
        metric_dict = self._calculate_metrics(preds, batch['labels'])
        self.log_dict(metric_dict, prog_bar=True, on_step=is_train, on_epoch=True)
        if is_train:
            return loss

    def training_step(self, batch, batch_idx):
        return self._step(batch, batch_idx, "train")

    def validation_step(self, batch, batch_idx, dataloader_idx=0):
        return self._step(batch, batch_idx, "val")

    def test_step(self, batch, batch_idx, dataloader_idx=0):
        return self._step(batch, batch_idx, "test")

    def _initialize_metrics(self, num_classes: int):
        self.precision_metric = pl.metrics.Precision(num_classes=num_classes).to(self.device)
        self.recall_metric = pl.metrics.Recall(num_classes=num_classes).to(self.device)
        self.accuracy_metric = pl.metrics.Accuracy().to(self.device)

    def _calculate_metrics(self, preds, labels, mode='val'):
        # Not required by all models. Only required for classification
        p = self.precision_metric(preds, labels)
        r = self.recall_metric(preds, labels)
        a = self.accuracy_metric(preds, labels)
        return {f'{mode}_precision': p, f'{mode}_recall': r, f'{mode}_acc': a}
        
