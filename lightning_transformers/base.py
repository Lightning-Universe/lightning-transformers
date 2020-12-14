from dataclasses import dataclass
from typing import List, Union, Optional

import pytorch_lightning as pl
import torch
from pytorch_lightning.utilities import rank_zero_only
from torch.optim import AdamW
from transformers import AutoTokenizer, AutoModelForSequenceClassification, AutoConfig, get_linear_schedule_with_warmup


# @seannaren TODO I did this for idea sake, not sure if it's the right thing going forward
# We could just keep all the params within the lightning module. But then should our base have default adam args?
# When do we want to add the adam args to the argparser? Having them in the default LitTransformer means anyone
# who overrides it, will also add them unless they override the argparse.
@dataclass
class TransformerAdamConfig:
    learning_rate: float = 2e-5
    adam_epsilon: float = 1e-8
    weight_decay: float = 0.0
    warmup_steps: int = 0

    @staticmethod
    def add_argparse_args(parser):
        parser.add_argument("--learning_rate", type=float, default=2e-5)
        parser.add_argument("--adam_epsilon", type=float, default=1e-8)
        parser.add_argument("--weight_decay", type=float, default=0.0)
        parser.add_argument("--warmup_steps", type=float, default=0)
        return parser

class LitTransformer(pl.LightningModule):
    def __init__(self,
                 model_name_or_path: Optional[str] = None,
                 tokenizer: Optional[AutoTokenizer] = None,
                 model_type: Optional[Union[AutoModelForSequenceClassification]] = None,
                 optim_config: Optional[TransformerAdamConfig] = None):
        super().__init__()
        self.save_hyperparameters()

        # We have to ensure that we only use rank 0 when downloading the model somehow.
        # This could cause issues otherwise.
        self.generate_config()

        self.model = model_type.from_pretrained(
            self.hparams.model_name_or_path, config=self.config
        )
        self.create_metrics()

    def generate_config(self):
        self.config = AutoConfig.from_pretrained(
            self.hparams.model_name_or_path,
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
        metric_dict = self.calculate_metrics(preds, batch['labels'])
        self.log_dict(metric_dict, prog_bar=True, on_step=False, on_epoch=True)
        self.log('val_loss', val_loss, prog_bar=True)

    def test_step(self, batch, batch_idx, dataloader_idx=0):
        idxs = batch.pop('idx')
        outputs = self(**batch)
        logits = outputs[0]
        preds = torch.argmax(logits, axis=1)


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
        optimizer = AdamW(
            optimizer_grouped_parameters,
            lr=self.hparams.optim_config.learning_rate,
            eps=self.hparams.optim_config.adam_epsilon
        )
        # @seannaren TODO this is going to be tricky. If we want to include total steps:
        # We'll need to calculate this via trainer arguments since we do not do this in lightning.
        # This will be needed as for most of the used HF schedulers they require knowing the total number of steps
        # I already have a function, but it requires a lot of leaking through the trainer etc, that might better in
        # main script like nate has done: https://github.com/nateraw/hf-text-classification/blob/main/train.py#L38
        scheduler = get_linear_schedule_with_warmup(
            optimizer, num_warmup_steps=10, num_training_steps=10 # TODO hardcoded till we expose
        )
        scheduler = {'scheduler': scheduler, 'interval': 'step', 'frequency': 1}
        return [optimizer], [scheduler]

    @rank_zero_only
    def save_pretrained(self, save_dir):
        self.hparams.save_dir = save_dir
        self.model.save_pretrained(self.hparams.save_dir)
        self.hparams.tokenizer.save_pretrained(self.hparams.save_dir)

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
