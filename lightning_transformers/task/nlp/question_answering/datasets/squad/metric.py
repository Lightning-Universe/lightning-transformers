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
            example_ids
        )
        predictions, references = self.postprocess_func(predictions=predictions)

        # with open('predictions.out', 'w') as f:
        #     for pred in predictions:
        #         f.write(f"{pred}\n")
        # with open('references.out', 'w') as f:
        #     for ref in references:
        #         f.write(f"{ref}\n")
        
        value = self.metric.compute(predictions=predictions, references=references)
        print(value)
        return value


# "/home/bjschre2/old_transformer/qa_ckpts/distilbert-base-cased_squad_64/"
# {'exact_match': 76.1968, 'f1': 84.4771}

# HF reference
# python -m pdb examples/question-answering/run_qa.py --model_name_or_path "/home/bjschre2/old_transformer/qa_ckpts/distilbert-base-cased_squad_64/" --dataset_name squad --do_eval --fp16 --max_seq_length 384 --doc_stride 128 --output_dir=/home/bjschre2/transformer-pruning/outputs/tmp

# Lightning
# python lightning-transformers/train.py task=nlp/question_answering dataset=nlp/question_answering/squad backbone.pretrained_model_name_or_path="/home/bjschre2/old_transformer/qa_ckpts/distilbert-base-cased_squad_64/" trainer.gpus=1 trainer.num_sanity_val_steps=-1 trainer.precision=16