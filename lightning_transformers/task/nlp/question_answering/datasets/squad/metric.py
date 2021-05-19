import torch
from datasets import load_metric
from pytorch_lightning.metrics import Metric

class SquadMetric(Metric):
    def __init__(self, postprocess_func):
        super().__init__()
        self.metric = load_metric("squad_v2")
        self.postprocess_func = postprocess_func
        self.add_state("preds", [])
        self.add_state("example_ids", [])

    def update(self, example_ids: torch.Tensor, preds: torch.Tensor):
        self.preds += preds
        self.example_ids += example_ids

    def compute(self):
        preds = {id: pred for id, pred in zip(self.example_ids, self.preds)}
        predictions, references = self.postprocess_func(preds)
        self.metric.compute(predictions, references, no_answer_threshold=1.0)
