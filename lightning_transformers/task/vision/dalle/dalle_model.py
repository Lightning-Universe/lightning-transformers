from typing import Any, Dict, List, Union

import hydra
import torch
from clip import clip, tokenize
from dalle_pytorch import DALLE, OpenAIDiscreteVAE

from lightning_transformers.core import TaskTransformer
from lightning_transformers.core.config import OptimizerConfig, SchedulerConfig
from lightning_transformers.core.instantiator import Instantiator


class DALLETransformer(TaskTransformer):

    def __init__(
        self,
        instantiator: Instantiator,
        backbone: Any,
        optimizer: OptimizerConfig,
        scheduler: SchedulerConfig,
    ):
        self.save_hyperparameters()
        vae = OpenAIDiscreteVAE()

        model: DALLE = hydra.utils.instantiate(backbone, vae=vae, num_text_tokens=vae.num_tokens)
        super().__init__(model=model, optimizer=optimizer, scheduler=scheduler, instantiator=instantiator)
        self.vae = vae
        for params in self.vae.parameters():
            params.requires_grad = False
        self.context_length = backbone.text_seq_len
        self._tokenizer = None

    @property
    def tokenizer(self):
        if self._tokenizer:
            return self._tokenizer
        return self.trainer.datamodule.tokenizer

    @tokenizer.setter
    def tokenizer(self, tokenizer):
        self._tokenizer = tokenizer

    def on_save_checkpoint(self, checkpoint: Dict[str, Any]):
        # Save tokenizer from datamodule for predict
        checkpoint["tokenizer"] = self.tokenizer

    def on_load_checkpoint(self, checkpoint: Dict[str, Any]) -> None:
        self.tokenizer = checkpoint["tokenizer"]

    def training_step(self, batch: Any, batch_idx: int) -> torch.Tensor:
        image, text = batch
        mask = torch.ones_like(text).bool().to(self.device)
        loss = self.model(text, image, mask, return_loss=True)
        return loss

    def test_step(self, batch: Any, batch_idx: int) -> torch.Tensor:
        return self.common_step(batch, "test")

    def validation_step(self, batch: Any, batch_idx: int) -> torch.Tensor:
        return self.common_step(batch, "validation")

    def common_step(self, batch: Any, prefix) -> torch.Tensor:
        image, text = batch
        mask = torch.ones_like(text).bool().to(self.device)
        loss = self.model(text, image, mask, return_loss=True)
        self.log(f"{prefix}_loss", loss, prog_bar=True, on_epoch=True, on_step=False)
        return loss

    def init(self):
        self.clip, self.preprocess = clip.load("ViT-B/32", device=self.device)

    def tokenize(self, texts: Union[str, List[str]], context_length: int = 77):
        clip._tokenizer = self.tokenizer
        return tokenize(texts, context_length)

    def generate_image(self, raw_text, num_samples=512):
        text = self.tokenize(raw_text, self.context_length).to(self.device)
        mask = torch.ones_like(text).bool().to(self.device)
        raw_image = self.model.generate_images(text, mask=mask)
        # text = clip.tokenize(["a diagram", "a dog", "a cat"]).to(device)
        #
        # with torch.no_grad():
        #     image_features = model.encode_image(image)
        #     text_features = model.encode_text(text)
        #
        #     logits_per_image, logits_per_text = model(image, text)
        #     probs = logits_per_image.softmax(dim=-1).cpu().numpy()
        return raw_image
