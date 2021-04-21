"""Root package info."""
import os

__ROOT_DIR__ = os.path.dirname(os.path.dirname(__file__))

__version__ = "0.1.0"
__author__ = "PyTorchLightning et al."
__author_email__ = "name@pytorchlightning.ai"
__license__ = 'Apache-2.0'
__copyright__ = f"Copyright (c) 2020-2020, {__author__}."
__homepage__ = "https://github.com/PyTorchLightning/lightning-transformers"
__docs__ = "PyTorch Lightning Transformers."
__long_doc__ = """
Flexible interface for high performance research using SOTA Transformers leveraging PyTorch Lightning,
 Transformers, and Hydra.

Transformers are increasingly popular for SOTA deep learning, gaining traction in NLP with BeRT based architectures
 more recently transcending into the world of Computer Vision and Audio Processing.

However, training and fine-tuning transformers at scale is not trivial and can vary from domain to domain requiring
 additional research effort, and significant engineering.

Lightning Transformers gives researchers a way to train HuggingFace Transformer models with all the features
 of PyTorch Lightning, while leveraging Hydra to provide composability of blocks and configs to focus on research.
"""
