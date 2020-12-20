from typing import Optional

import pytorch_lightning as pl
from hydra.utils import get_class, instantiate
from omegaconf import DictConfig
from pytorch_lightning.utilities import rank_zero_only
from transformers import AutoConfig


class LitTransformer(pl.LightningModule):
    def __init__(self,
                 pretrained_model_name_or_path: str,
                 model_type: str,
                 optim: DictConfig,
                 scheduler: Optional[DictConfig] = None,
                 **kwargs):
        super().__init__()
        # Resolve the bug in Lightning save_hyperparameters
        optim.lr = optim.lr  # todo wat

        self.save_hyperparameters()
        self.tokenizer = None  # todo set via data module

        # We have to ensure that we only use rank 0 when downloading the model somehow.
        # This could cause issues otherwise.
        self.generate_config()

        self.model = get_class(self.hparams.model_type).from_pretrained(
            pretrained_model_name_or_path=self.hparams.pretrained_model_name_or_path,
            config=self.config
        )

    def generate_config(self):
        self.config = AutoConfig.from_pretrained(
            self.hparams.pretrained_model_name_or_path,
        )

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
        self.tokenizer.save_pretrained(self.hparams.save_dir)
