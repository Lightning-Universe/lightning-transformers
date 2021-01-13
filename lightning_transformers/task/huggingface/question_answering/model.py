from typing import Any, Dict

import torch

from lightning_transformers.core.huggingface import HFTransformer


class QuestionAnsweringTransformer(HFTransformer):
    def training_step(self, batch: Any, batch_idx: int) -> torch.Tensor:
        outputs = self.model(**batch)
        loss = outputs[0]
        self.log("train_loss", loss)
        return loss

    def validation_step(self, batch: Any, batch_idx: int, dataloader_idx: int = 0) -> Dict[str, torch.Tensor]:
        outputs = self.model(**batch)
        loss = outputs[0]
        self.log("val_loss", loss, prog_bar=True, sync_dist=True)
        return loss

    def test_step(self, batch: Any, batch_idx: int, dataloader_idx: int = 0):
        outputs = self.model(**batch)
        loss, logits = outputs[:2]
        # preds = torch.argmax(logits, dim=1)
        # metric_dict = self.compute_metrics(batch, preds)
        # self.log_dict(metric_dict, prog_bar=True, on_step=False, on_epoch=True, sync_dist=True)
        return loss

    # TODO?
    # def validation_epoch_end(self, outputs):
    #     start_logits = []
    #     end_logits = []
    #     for o in outputs:
    #         s_logits = o["start_logits"]
    #         e_logits = o["end_logits"]
    #         if self.trainer.use_ddp:
    #             s_logits = self.all_gather(s_logits)
    #             e_logits = self.all_gather(e_logits)
    #         start_logits.append(s_logits)
    #         end_logits.append(e_logits)
    #     start_logits = torch.cat(start_logits, dim=0)
    #     end_logits = torch.cat(end_logits, dim=0)
    #     self._calculate_metrics((start_logits, end_logits))

    # def configure_metrics(self, stage: str):
    #    pass

    # def compute_metrics(self, preds, labels, mode: str = "val"):
    #    preds = self.trainer.data_module.post_process_function(preds)
    #    return self.metric.compute(predictions=preds, references=labels)
