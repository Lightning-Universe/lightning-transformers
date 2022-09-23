# Copyright The PyTorch Lightning team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
from pathlib import Path
from typing import IO, Any, Callable, Dict, Optional, Tuple, Type, Union

import pytorch_lightning as pl
import torch
import transformers
from pytorch_lightning.utilities import rank_zero_warn
from transformers import AutoConfig, Pipeline, PreTrainedTokenizerBase
from transformers import pipeline as hf_transformers_pipeline
from transformers.models.auto.auto_factory import _BaseAutoModelClass

from lightning_transformers.utilities.deepspeed import enable_transformers_pretrained_deepspeed_sharding
from lightning_transformers.utilities.imports import _ACCELERATE_AVAILABLE

if _ACCELERATE_AVAILABLE:
    from accelerate import load_checkpoint_and_dispatch


class TaskTransformer(pl.LightningModule):
    """Base class for task specific transformers, wrapping pre-trained language models for downstream tasks. The
    API is built on top of AutoModel and AutoConfig, provided by HuggingFace.

    see: https://huggingface.co/transformers/model_doc/auto.html

    Args:
        downstream_model_type: The AutoModel downstream model type.
            See https://huggingface.co/transformers/model_doc/auto.html
        pretrained_model_name_or_path: Huggingface model to use if backbone config not passed.
        tokenizer: The pre-trained tokenizer.
        pipeline_kwargs: Arguments required for the HuggingFace inference pipeline class.
    """

    def __init__(
        self,
        downstream_model_type: Type[_BaseAutoModelClass],
        pretrained_model_name_or_path: Optional[str] = None,
        tokenizer: Optional[PreTrainedTokenizerBase] = None,
        pipeline_kwargs: Optional[dict] = None,
        load_weights: bool = True,
        deepspeed_sharding: bool = False,
        **model_data_kwargs,
    ) -> None:
        super().__init__()
        self.save_hyperparameters()
        self.load_weights = load_weights
        self.model_data_kwargs = model_data_kwargs
        self.downstream_model_type = downstream_model_type
        self.deepspeed_sharding = deepspeed_sharding
        self.pretrained_model_name_or_path = pretrained_model_name_or_path
        if not self.deepspeed_sharding:
            self.initialize_model(self.pretrained_model_name_or_path)
        self._tokenizer = tokenizer  # necessary for hf_pipeline
        self._hf_pipeline = None
        self._hf_pipeline_kwargs = pipeline_kwargs or {}

    def initialize_model(self, pretrained_model_name_or_path: str):
        """create and initialize the model to use with this task,

        Feel free to overwrite this method if you are initializing the model in a different way
        """
        if self.load_weights:
            self.model = self.downstream_model_type.from_pretrained(
                pretrained_model_name_or_path, **self.model_data_kwargs
            )
        else:
            config = AutoConfig.from_pretrained(pretrained_model_name_or_path, **self.model_data_kwargs)
            self.model = self.downstream_model_type.from_config(config)

    def configure_optimizers(self) -> Dict:
        rank_zero_warn(
            "You haven't specified an optimizer or lr scheduler. "
            "Defaulting to AdamW with an lr of 1e-5 and linear warmup for 10% of steps. "
            "To change this, override ``configure_optimizers`` in  TransformerModule."
        )
        optimizer = torch.optim.AdamW(self.parameters(), lr=1e-5)
        num_training_steps, num_warmup_steps = self.compute_warmup(
            num_training_steps=-1,
            num_warmup_steps=0.1,
        )
        scheduler = transformers.get_linear_schedule_with_warmup(
            optimizer, num_warmup_steps=num_warmup_steps, num_training_steps=num_training_steps
        )
        return {
            "optimizer": optimizer,
            "lr_scheduler": {"scheduler": scheduler, "interval": "step", "frequency": 1},
        }

    @property
    def num_training_steps(self) -> int:
        return self.trainer.estimated_stepping_batches

    def compute_warmup(self, num_training_steps: int, num_warmup_steps: Union[int, float]) -> Tuple[int, int]:
        if num_training_steps < 0:
            # less than 0 specifies to infer number of training steps
            num_training_steps = self.num_training_steps
        if isinstance(num_warmup_steps, float):
            # Convert float values to percentage of training steps to use as warmup
            num_warmup_steps *= num_training_steps
        return num_training_steps, num_warmup_steps

    def setup(self, stage: Optional[str] = None) -> None:
        self.configure_metrics(stage)
        if self.deepspeed_sharding and not hasattr(self, "model"):
            enable_transformers_pretrained_deepspeed_sharding(self)
            self.initialize_model(self.pretrained_model_name_or_path)

    def configure_metrics(self, stage: str) -> Optional[Any]:
        """Override to configure metrics for train/validation/test.

        This is called on fit start to have access to the data module, and initialize any data specific metrics.
        """
        pass

    @property
    def tokenizer(self) -> Optional[PreTrainedTokenizerBase]:
        if (
            self._tokenizer is None
            and hasattr(self, "trainer")  # noqa: W503
            and hasattr(self.trainer, "datamodule")  # noqa: W503
            and hasattr(self.trainer.datamodule, "tokenizer")  # noqa: W503
        ):
            self._tokenizer = self.trainer.datamodule.tokenizer
        return self._tokenizer

    @tokenizer.setter
    def tokenizer(self, tokenizer: PreTrainedTokenizerBase) -> None:
        self._tokenizer = tokenizer

    @property
    def hf_pipeline_task(self) -> Optional[str]:
        """Override to define what HuggingFace pipeline task to use.

        Returns: Optional string to define what pipeline task to use.
        """
        return None

    @property
    def hf_pipeline(self) -> Pipeline:
        if self._hf_pipeline is None:
            if self.hf_pipeline_task is not None:
                self._hf_pipeline = hf_transformers_pipeline(
                    task=self.hf_pipeline_task, model=self.model, tokenizer=self.tokenizer, **self._hf_pipeline_kwargs
                )
            else:
                raise RuntimeError("No task was defined for this model. Try overriding `hf_pipeline_task`")
        return self._hf_pipeline

    @hf_pipeline.setter
    def hf_pipeline(self, pipeline: Pipeline) -> None:
        self._hf_pipeline = pipeline

    def hf_predict(self, *args, **kwargs) -> Any:
        return self.hf_pipeline(*args, **kwargs)

    @classmethod
    def load_from_checkpoint(
        cls,
        checkpoint_path: Union[str, IO],
        map_location: Optional[Union[Dict[str, str], str, torch.device, int, Callable]] = None,
        hparams_file: Optional[str] = None,
        strict: bool = True,
        hf_pipeline_kwargs: Optional[Dict] = None,
        **kwargs,
    ):
        model: TaskTransformer = super().load_from_checkpoint(checkpoint_path, map_location, hparams_file, strict)
        # update model with hf_pipeline_kwargs override
        if hf_pipeline_kwargs is not None:
            model._hf_pipeline_kwargs.update(hf_pipeline_kwargs)
        return model

    def load_checkpoint_and_dispatch(self, *args, **kwargs) -> None:
        """Use when loading checkpoint via accelerate for large model support.

        Useful for when loading sharded checkpoints.
        """
        self.model = load_checkpoint_and_dispatch(self.model, *args, **kwargs)

    @property
    def hf_device_map(self) -> Dict:
        """
        Returns: Device Map as defined when using `load_checkpoint_and_dispatch`.
        """
        return self.model.hf_device_map

    def save_hf_checkpoint(self, path: Union[str, Path]) -> None:
        """Save the model using the original HF AutoModel.

        This is useful for when you'd like to export the model to the hub.
        Args:
            path: Path to save the model to.
        """
        self.model.save_pretrained(path)
