from dataclasses import dataclass
from typing import Union, Optional
from hydra.utils import get_class, instantiate
import pytorch_lightning as pl
from pytorch_lightning.utilities import rank_zero_only
from torch.optim import AdamW
from transformers import AutoTokenizer, AutoModelForSequenceClassification, AutoConfig


class LitTransformer(pl.LightningModule):
    def __init__(self, name: Optional[str] = None, model_type: str = None, optim = None, scheduler = None):
        super().__init__()
        # Resolve the bug in Lightning save_hyperparameters
        optim.lr = optim.lr
        
        self.save_hyperparameters()

        # We have to ensure that we only use rank 0 when downloading the model somehow.
        # This could cause issues otherwise.
        self.generate_config()

        self.model = get_class(model_type).from_pretrained(
            self.hparams.name, config=self.config
        )
        self.create_metrics()

    def generate_config(self):
        self.config = AutoConfig.from_pretrained(
            self.hparams.name,
        )

    def create_metrics(self):
        self.precision_metric = pl.metrics.Precision(num_classes=len(self.hparams.label2id))
        self.recall_metric = pl.metrics.Recall(num_classes=len(self.hparams.label2id))
        self.accuracy_metric = pl.metrics.Accuracy()

    def calculate_metrics(self, preds, labels, mode='val'):
        # Not required by all models. Only required for classification
        p = self.precision_metric(preds, labels)
        r = self.recall_metric(preds, labels)
        a = self.accuracy_metric(preds, labels)
        return {f'{mode}_precision': p, f'{mode}_recall': r, f'{mode}_acc': a}

    def forward(self, **inputs):
        return self.model(**inputs)

    def training_step(self, batch, batch_idx):
        outputs = self(**batch)
        loss = outputs[0]
        self.log('train_loss', loss)
        return loss

    def validation_step(self, batch, batch_idx, dataloader_idx=0):
        outputs = self(**batch)
        val_loss, logits = outputs[:2]
        preds = torch.argmax(logits, axis=1)
        metric_dict = self.calculate_metrics(batch, preds)
        self.log_dict(metric_dict, prog_bar=True, on_step=False, on_epoch=True)
        self.log('val_loss', val_loss, prog_bar=True)

    def test_step(self, batch, batch_idx, dataloader_idx=0):
        outputs = self(**batch)
        logits = outputs[0]
        preds = torch.argmax(logits, axis=1)
        metric_dict = self.calculate_metrics(batch, preds)
        self.log_dict(metric_dict, prog_bar=True, on_step=False, on_epoch=True)
        self.log('val_loss', val_loss, prog_bar=True)

    def configure_optimizers(self):
        """Prepare optimizer and schedule (linear warmup and decay)"""

        # TODO We should offer defaults, and allow the user to override the module like a normal lightning module.
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
        optimizer = instantiate(self.hparams.optim, optimizer_grouped_parameters)
        scheduler = instantiate(self.hparams.scheduler, optimizer)
        scheduler = {'scheduler': scheduler, 'interval': 'step', 'frequency': 1}
        return [optimizer], [scheduler]

    @rank_zero_only
    def save_pretrained(self, save_dir):
        self.hparams.save_dir = save_dir
        self.model.save_pretrained(self.hparams.save_dir)
        self.hparams.tokenizer.save_pretrained(self.hparams.save_dir)