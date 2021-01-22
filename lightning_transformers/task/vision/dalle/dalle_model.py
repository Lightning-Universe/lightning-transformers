from typing import Any

import hydra
import torch
from dalle_pytorch import DALLE

from lightning_transformers.core.hydra_model import HydraTaskTransformer
from lightning_transformers.task.vision.dalle import VQVAE
from lightning_transformers.task.vision.dalle.clip import clip
from lightning_transformers.task.vision.dalle.clip.clip import tokenize
from lightning_transformers.task.vision.dalle.clip.simple_tokenizer import SimpleTokenizer


class DALLETransformer(HydraTaskTransformer):
    def __init__(
        self, vae_path: str, tokenizer_path: str, backbone: Any, optimizer: Any, scheduler: Any, **config_data_args
    ):
        super().__init__(optimizer, scheduler)
        self.save_hyperparameters()
        self.vae = VQVAE.load_from_checkpoint(vae_path)
        for params in self.vae.parameters():
            params.requires_grad = False
        self.tokenizer = SimpleTokenizer(tokenizer_path)
        self.model: DALLE = hydra.utils.instantiate(
            backbone, vae=self.vae.model, num_text_tokens=len(self.tokenizer.encoder)
        )
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
