from typing import List
import torch
from transformers import (
    AutoModelForSequenceClassification,
    AutoModelForQuestionAnswering,
    AutoTokenizer,
)

from lightning_transformers.core.base import LitTransformer


class LitLanguageModelingTransformer(LitTransformer):
    def __init__(self, model_name_or_path: str, tokenizer: AutoTokenizer, optim_config):
        super().__init__(
            model_name_or_path=model_name_or_path,
            tokenizer=tokenizer,
            optim_config=optim_config,
            model_type=AutoModelForSequenceClassification
        )


class LitMultipleChoiceTransformer(LitTransformer):
    def __init__(
            self,
            model_name_or_path: str,
            tokenizer: AutoTokenizer,
            optim_config):
        super().__init__(
            model_name_or_path=model_name_or_path,
            tokenizer=tokenizer,
            optim_config=optim_config,
            model_type=AutoModelForSequenceClassification
        )


class LitQuestionAnsweringTransformer(LitTransformer):

    def test_step(self, batch, batch_idx, dataloader_idx=0):
        outputs = self(**batch)
        logits = outputs[0]
        preds = torch.argmax(logits, axis=1)
        metric_dict = self.calculate_metrics(batch, preds)
        self.log_dict(metric_dict, prog_bar=True, on_step=False, on_epoch=True)

    def create_metrics(self):
        pass

    def training_step(self, batch, batch_idx):
        outputs = self(**batch)
        loss = outputs[0]
        self.log('train_loss', loss)
        return loss

    def validation_step(self, batch, batch_idx, dataloader_idx=0):
        outputs = self(**batch)
        val_loss, logits = outputs[:2]
        preds = torch.argmax(logits, axis=1)
        self.log('val_loss', val_loss, prog_bar=True)
        return {'start_logits': outputs.start_logits, "end_logits": outputs.end_logits}

    def validation_epoch_end(self, outputs):
        # wip
        if False:
            start_logits = []
            end_logits = []
            for o in outputs:
                s_logits = o["start_logits"]
                e_logits = o["end_logits"]
                if self.trainer.use_ddp:
                    s_logits = self.all_gather(s_logits)
                    e_logits = self.all_gather(e_logits)
                start_logits.append(s_logits)
                end_logits.append(e_logits)
            start_logits = torch.cat(start_logits, dim=0)
            end_logits = torch.cat(end_logits, dim=0)
            self.calculate_metrics((start_logits, end_logits))

    def test_step(self, batch, batch_idx, dataloader_idx=0):
        outputs = self(**batch)
        logits = outputs[0]
        preds = torch.argmax(logits, axis=1)


class LitTextClassificationTransformer(LitTransformer):
    def __init__(
            self,
            model_name_or_path: str,
            tokenizer: AutoTokenizer,
            optim_config,
            num_classes: int):
        self.num_classes = num_classes
        super().__init__(
            model_name_or_path=model_name_or_path,
            tokenizer=tokenizer,
            optim_config=optim_config,
            model_type=AutoModelForSequenceClassification
        )

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
        self.precision_metric = pl.metrics.Precision(num_classes=self.num_classes)
        self.recall_metric = pl.metrics.Recall(num_classes=self.num_classes)
        self.accuracy_metric = pl.metrics.Accuracy()


class LitTextGenerationTransformer(LitTransformer):
    def __init__(
            self,
            model_name_or_path: str,
            tokenizer: AutoTokenizer,
            optim_config):
        super().__init__(
            model_name_or_path=model_name_or_path,
            tokenizer=tokenizer,
            optim_config=optim_config,
            model_type=AutoModelForSequenceClassification
        )


class LitTokenClassificationTransformer(LitTransformer):
    def __init__(
            self,
            model_name_or_path: str,
            tokenizer: AutoTokenizer,
            optim_config):
        super().__init__(
            model_name_or_path=model_name_or_path,
            tokenizer=tokenizer,
            optim_config=optim_config,
            model_type=AutoModelForSequenceClassification
        )
