import torch
from typing import Optional
from omegaconf import DictConfig
import pytorch_lightning as pl
from lightning_transformers.core.model import LitTransformer


class LitLanguageModelingTransformer(LitTransformer):

    def training_step(self, batch, batch_idx):
        outputs = self(*batch)
        import pdb; pdb.set_trace()
        
