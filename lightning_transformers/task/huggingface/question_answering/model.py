from functools import partial
from typing import Any, Dict

import torch

from lightning_transformers.core.huggingface import HFTransformer
from lightning_transformers.task.huggingface.question_answering import QuestionAnsweringTransformerDataModule
from lightning_transformers.task.huggingface.question_answering.metric import SquadMetric


class QuestionAnsweringTransformer(HFTransformer):
    def training_step(self, batch: Any, batch_idx: int) -> torch.Tensor:
        outputs = self.model(**batch)
        loss = outputs[0]
        self.log("train_loss", loss)
        return loss

    def validation_step(self, batch: Any, batch_idx: int, dataloader_idx: int = 0) -> Dict[str, torch.Tensor]:
        # todo: however is needed for metrics computation eventually...
        batch.pop("offset_mapping")
        outputs = self.model(**batch)
        example_ids = batch["example_id"]
        loss = outputs[0]
        self.log("val_loss", loss, prog_bar=True, sync_dist=True)
        self.metric(example_ids, outputs)
        return loss

    def test_step(self, batch: Any, batch_idx: int, dataloader_idx: int = 0):
        outputs = self.model(**batch)
        loss, logits = outputs[:2]
        return loss

    def configure_metrics(self, stage: str):
        dataset: QuestionAnsweringTransformerDataModule = self.trainer.datamodule
        validation_dataset = dataset.ds["validation"]
        original_validation_dataset = dataset.ds["validation_original"]
        postprocess_func = partial(
            dataset.postprocess_func,
            dataset=dataset.ds,
            validation_dataset=validation_dataset,
            original_validation_dataset=original_validation_dataset,
        )
        self.metric = SquadMetric(postprocess_func=postprocess_func)
