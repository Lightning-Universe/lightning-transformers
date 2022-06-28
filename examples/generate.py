"""Example to use a large language model to generate text.

First, Transformers will download the sharded weights (only some models are compatible as a result). Accelerate will
then automatically place weights on appropriate devices to fit into CPU and GPU memory.
"""

import torch
from accelerate import init_empty_weights
from transformers import AutoTokenizer

from lightning_transformers.task.nlp.language_modeling import LanguageModelingTransformer

with init_empty_weights():
    model = LanguageModelingTransformer(
        pretrained_model_name_or_path="EleutherAI/gpt-j-6B",
        tokenizer=AutoTokenizer.from_pretrained("EleutherAI/gpt-j-6B"),
        low_cpu_mem_usage=True,
        device_map="auto",
    )

output = model.generate("Hello, my name is", device=torch.device("cuda"))
print(model.tokenizer.decode(output[0].tolist()))
