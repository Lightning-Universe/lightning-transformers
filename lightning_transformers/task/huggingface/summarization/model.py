from dataclasses import dataclass
from typing import Any

import torch

from lightning_transformers.core.huggingface.seq2seq.model import Seq2SeqTransformer
from lightning_transformers.task.huggingface.summarization.metric import RougeMetric


@dataclass
class SummarizationTransformerMetricsConfig:
    use_stemmer: bool = True
    rouge_newline_sep: bool = True
    return_precision_and_recall: bool = True


class SummarizationTransformer(Seq2SeqTransformer):
    def __init__(
        self,
        *args,
        metrics_cfg: SummarizationTransformerMetricsConfig = SummarizationTransformerMetricsConfig(),
        **kwargs
    ):
        super().__init__(*args, **kwargs)
        self.metrics = None
        self.metrics_cfg = metrics_cfg

    @property
    def task(self) -> str:
        return "summarization"

    def common_step(self, prefix: str, batch: Any) -> torch.Tensor:
        outputs = self.model(**batch)
        loss, logits = outputs[:2]
        self.compute_metrics(logits, batch["labels"])
        return loss

    def compute_metrics(self, pred, labels):
        pred = torch.argmax(pred, dim=2)
        pred_lns, tgt_lns = self.decode(pred, labels)
        result = self.rouge(pred_lns, tgt_lns)
        self.log_dict(result, on_step=False, on_epoch=True, prog_bar=True)

    def configure_metrics(self, stage: str):
        self.rouge = RougeMetric(
            return_precision_and_recall=self.metrics_cfg.return_precision_and_recall,
            rouge_newline_sep=self.metrics_cfg.rouge_newline_sep,
            use_stemmer=self.metrics_cfg.use_stemmer,
        )
