import torch
from datasets import load_metric
from torchmetrics import Metric


class SquadMetric(Metric):
    def __init__(self, postprocess_func, example_id_strings):
        super().__init__(compute_on_step=False)
        self.metric = load_metric("squad")
        self.postprocess_func = postprocess_func
        self.example_id_strings = example_id_strings
        self.add_state("start_logits", [])
        self.add_state("end_logits", [])
        self.add_state("example_ids", [])

    def update(self, example_ids: torch.Tensor, start_logits: torch.Tensor, end_logits: torch.Tensor):
        self.example_ids += example_ids
        self.start_logits += start_logits
        self.end_logits += end_logits

    def compute(self):
        reverse_lookup = {i: s for s, i in self.example_id_strings.items()}
        example_ids = [reverse_lookup[i.item()] for i in self.example_ids]
        predictions = (
            torch.stack(self.start_logits).cpu().numpy(),
            torch.stack(self.end_logits).cpu().numpy(),
            example_ids,
        )
        predictions, references = self.postprocess_func(predictions=predictions)
        value = self.metric.compute(predictions=predictions, references=references)
        return value
