from typing import Any, Callable, Iterable, List, Tuple

import torch
from pytorch_lightning import _logger as log

from lightning_transformers.core.huggingface import HFTransformer


class Seq2SeqTransformer(HFTransformer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.metrics = {}

    def training_step(self, batch: Any, batch_idx: int) -> torch.Tensor:
        outputs = self.model(**batch)
        loss = outputs[0]
        self.log("train_loss", loss)
        return loss

    def common_step(self, prefix: str, batch: Any) -> torch.Tensor:
        outputs = self.model(**batch)
        loss, logits = outputs[:2]
        return loss

    def validation_step(self, batch: Any, batch_idx: int, dataloader_idx: int = 0) -> torch.Tensor:
        return self.common_step("val", batch)

    def test_step(self, batch: Any, batch_idx: int, dataloader_idx: int = 0) -> torch.Tensor:
        return self.common_step("test", batch)

    def on_fit_start(self):
        self.initialize_model_specific_parameters()

    def initialize_model_specific_parameters(self):
        task_specific_params = self.model.config.task_specific_params

        if task_specific_params is not None:
            pars = task_specific_params.get(self.task, {})
            log.info(f"Setting model params for {self.task}:\n {pars}")
            self.model.config.update(pars)

    @property
    def task(self) -> str:
        raise NotImplementedError

    def decode(self, preds: torch.Tensor, labels: torch.Tensor) -> Tuple[List[str], List[str]]:
        pred_str = self.tokenizer.batch_decode(preds, skip_special_tokens=True)
        label_str = self.tokenizer.batch_decode(labels, skip_special_tokens=True)
        pred_str = self.lmap(str.strip, pred_str)
        label_str = self.lmap(str.strip, label_str)
        return pred_str, label_str

    def lmap(self, f: Callable, x: Iterable) -> List:
        """list(map(f, x))"""
        return list(map(f, x))
