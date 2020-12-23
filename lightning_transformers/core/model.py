from typing import Optional, Any

import pytorch_lightning as pl
from hydra.utils import get_class, instantiate
from omegaconf import DictConfig
from transformers import AutoConfig


class LitTransformer(pl.LightningModule):

    def __init__(self,
                 model: Any,
                 optim: DictConfig,
                 scheduler: DictConfig,
                 tokenizer: Any = None):
        super().__init__()
        self.tokenizer = tokenizer
        self.model = model
        self.scheduler = scheduler
        self.optim = optim

    def configure_optimizers(self):
        """Prepare optimizer and schedule (linear warmup and decay)"""
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
        optimizer = instantiate(self.optim, optimizer_grouped_parameters)
        scheduler = instantiate(self.scheduler, optimizer)
        scheduler = {'scheduler': scheduler, 'interval': 'step', 'frequency': 1}
        return [optimizer], [scheduler]

    def forward(self, **inputs):
        return self.model(**inputs)


class LitAutoModelTransformer(LitTransformer):
    def __init__(self,
                 downstream_model_type: str,
                 backbone: DictConfig,
                 optim: DictConfig,
                 scheduler: Optional[DictConfig] = None,
                 **kwargs):
        # Resolve the bug in Lightning save_hyperparameters
        optim.lr = optim.lr  # todo Resolve this bug in Lightning directly

        self.save_hyperparameters()

        # We have to ensure that we only use rank 0 when downloading the model somehow.
        # This could cause issues otherwise.
        self.generate_config()

        model = get_class(self.hparams.downstream_model_type).from_pretrained(
            pretrained_model_name_or_path=self.hparams.backbone.pretrained_model_name_or_path,
            config=self.config
        )
        super().__init__(
            model=model,
            optim=optim,
            scheduler=scheduler
        )

    def generate_config(self):
        self.config = AutoConfig.from_pretrained(
            self.hparams.backbone.pretrained_model_name_or_path,
        )
