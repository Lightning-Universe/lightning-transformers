import torch
from transformers import AutoTokenizer

from lightning_transformers.task.nlp.language_modeling import LanguageModelingTransformer

model = LanguageModelingTransformer(
    pretrained_model_name_or_path="bigscience/bloom",
    tokenizer=AutoTokenizer.from_pretrained("bigscience/bloom"),
    low_cpu_mem_usage=True,
    device_map="auto",
)

output = model.generate("Hello, my name is", device=torch.device("cuda"))
print(model.tokenizer.decode(output[0].tolist()))
