"""Root package info."""
import os

__ROOT_DIR__ = os.path.dirname(os.path.dirname(__file__))

__version__ = "0.2.3"
__author__ = "PyTorchLightning et al."
__author_email__ = "name@pytorchlightning.ai"
__license__ = "Apache-2.0"
__copyright__ = f"Copyright (c) 2020-2020, {__author__}."
__homepage__ = "https://github.com/PyTorchLightning/lightning-transformers"
__docs__ = "PyTorch Lightning Transformers."
__long_doc__ = """
Lightning Transformers provides `LightningModules`, `LightningDataModules` and `Strategies` to use
HuggingFace Transformers with the PyTorch Lightning Trainer.
"""
