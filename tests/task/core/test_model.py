import torch

from lightning_transformers.core import LitTransformer


def test_lit_transformer():
    LitTransformer(torch.nn.Linear(1, 1))
