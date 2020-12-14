from typing import Union

import pytorch_lightning as pl
import torch
from pytorch_lightning.utilities import rank_zero_only
from torch.optim import AdamW
from transformers import (
    AutoConfig,
    AutoTokenizer,
)
from transformers import (
    AutoModelForSequenceClassification,
)
from transformers import (
    get_linear_schedule_with_warmup
)


class LitTransformer(pl.LightningModule):
    def __init__(self,
                 model_name_or_path: str,
                 tokenizer: AutoTokenizer,
                 model_type: Union[AutoModelForSequenceClassification, AutoModelForSequenceClassification]):
        super().__init__()
        self.save_hyperparameters()

        # We have to ensure that we only use rank 0 when downloading the model somehow.
        # This could cause issues otherwise.
        self.config = AutoConfig.from_pretrained(model_name_or_path)
        self.model = model_type.from_pretrained(
            pretrained_model_name_or_path=self.hparams.model_name_or_path,
            config=self.config
        )
        self.tokenizer = tokenizer

        self.create_metrics()

    def create_metrics(self):
        raise NotImplementedError

    def log_metrics(self, preds, labels, mode='val'):
        pass

    def forward(self, **inputs):
        return self.model(**inputs)

    def configure_optimizers(self):
        """Prepare optimizer and schedule (linear warmup and decay)"""

        # TODO We should offer defaults, and allow the user to override the module like a normal lightning module.
        no_decay = ["bias", "LayerNorm.weight"]
        optimizer_grouped_parameters = [
            {
                "params": [p for n, p in self.model.named_parameters() if not any(nd in n for nd in no_decay)],
                "weight_decay": self.hparams.weight_decay,
            },
            {
                "params": [p for n, p in self.model.named_parameters() if any(nd in n for nd in no_decay)],
                "weight_decay": 0.0,
            },
        ]
        optimizer = AdamW(optimizer_grouped_parameters, lr=self.hparams.learning_rate, eps=self.hparams.adam_epsilon)
        scheduler = get_linear_schedule_with_warmup(
            optimizer, num_warmup_steps=self.hparams.warmup_steps, num_training_steps=self.total_steps
        )
        scheduler = {'scheduler': scheduler, 'interval': 'step', 'frequency': 1}
        return [optimizer], [scheduler]

    @rank_zero_only
    def save_pretrained(self, save_dir):
        self.hparams.save_dir = save_dir
        self.model.save_pretrained(self.hparams.save_dir)
        self.tokenizer.save_pretrained(self.hparams.save_dir)

    @staticmethod
    def add_argparse_args(parser):
        parser.add_argument("--model_name_or_path", type=str,
                            help="Path to pretrained model or model identifier from huggingface.co/models")
        parser.add_argument("--config_name", type=str, default=None,
                            help="Pretrained config name or path if not the same as model_name")
        parser.add_argument("--tokenizer_name", type=str, default=None,
                            help="Pretrained tokenizer name or path if not the same as model_name")
        parser.add_argument("--cache_dir", type=str,
                            help="Path to directory to store the pretrained models downloaded from huggingface.co")
        return parser


class LitTextClassificationTransformer(LitTransformer):

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

    def create_metrics(self):
        self.precision_metric = pl.metrics.Precision(num_classes=len(self.hparams.label2id))
        self.recall_metric = pl.metrics.Recall(num_classes=len(self.hparams.label2id))
        self.accuracy_metric = pl.metrics.Accuracy()
