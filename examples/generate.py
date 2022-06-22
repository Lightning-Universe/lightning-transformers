"""Example to use a large language model to generate text. Accelerate will automatically place weights on
appropriate devices to fit into CPU and GPU memory.

Please download the sharded weights manually before running:

git clone https://huggingface.co/sgugger/sharded-gpt-j-6B
cd sharded-gpt-j-6B
git-lfs install
git pull
"""

import torch
from accelerate import init_empty_weights
from transformers import AutoTokenizer

from lightning_transformers.task.nlp.language_modeling import LanguageModelingTransformer

with init_empty_weights():
    model = LanguageModelingTransformer(
        pretrained_model_name_or_path="EleutherAI/gpt-j-6B",
        tokenizer=AutoTokenizer.from_pretrained("EleutherAI/gpt-j-6B"),
    )

model.load_checkpoint_and_dispatch("sharded-gpt-j-6B", device_map="auto", no_split_module_classes=["GPTJBlock"])

output = model.generate("Hello, my name is", device=torch.device("cuda"))
print(model.tokenizer.decode(output[0].tolist()))
