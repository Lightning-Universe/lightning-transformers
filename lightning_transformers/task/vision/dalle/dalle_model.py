from typing import Any

import hydra
import torch
from dalle_pytorch import DALLE

from lightning_transformers.core import TaskTransformer
from lightning_transformers.core.config import OptimizerConfig, SchedulerConfig
from lightning_transformers.core.instantiator import Instantiator
from lightning_transformers.task.vision.dalle import VQVAE
from lightning_transformers.task.vision.dalle.clip import clip
from lightning_transformers.task.vision.dalle.clip.clip import tokenize
from lightning_transformers.task.vision.dalle.clip.simple_tokenizer import SimpleTokenizer


class DALLETransformer(TaskTransformer):
    def __init__(
        self,
        instantiator: Instantiator,
        vae_path: str,
        tokenizer_path: str,
        backbone: Any,
        optimizer: OptimizerConfig,
        scheduler: SchedulerConfig,
    ):
        self.save_hyperparameters()
        vae = VQVAE.load_from_checkpoint(vae_path)
        self.model: DALLE = hydra.utils.instantiate(
            backbone, vae=vae.model, num_text_tokens=len(self.tokenizer.encoder)
        )
        self.tokenizer = SimpleTokenizer(tokenizer_path)
        super().__init__(model=self.model, optimizer=optimizer, scheduler=scheduler, instantiator=instantiator)
        self.vae = vae
        for params in self.vae.parameters():
            params.requires_grad = False
        self.context_length = backbone.text_seq_len

    def training_step(self, batch: Any, batch_idx: int) -> torch.Tensor:
        image, text = batch
        text = tokenize(text, self.tokenizer, self.context_length).to(self.device)
        mask = torch.ones_like(text).bool().to(self.device)
        loss = self.model(text, image, mask, return_loss=True)
        return loss

    def test_step(self, batch: Any, batch_idx: int) -> torch.Tensor:
        return self.common_step(batch, "test")

    def validation_step(self, batch: Any, batch_idx: int) -> torch.Tensor:
        return self.common_step(batch, "validation")

    def common_step(self, batch: Any, prefix) -> torch.Tensor:
        image, text = batch
        text = tokenize(text, self.tokenizer, self.context_length).to(self.device)
        mask = torch.ones_like(text).bool().to(self.device)
        loss = self.model(text, image, mask, return_loss=True)
        self.log(f"{prefix}_loss", loss, prog_bar=True, on_epoch=True, on_step=False)
        return loss

    def init(self):
        self.clip, self.preprocess = clip.load("ViT-B/32", device="cpu")

    def generate_image(self, raw_text, num_samples=512):
        text = tokenize(raw_text, self.tokenizer, self.context_length).to(self.device)
        mask = torch.ones_like(text).bool().to(self.device)
        raw_image = self.model.generate_images(text, mask=mask)
        return raw_image
