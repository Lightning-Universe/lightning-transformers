from typing import IO, TYPE_CHECKING, Any, Callable, Dict, Optional, Type, Union

import torch
from hydra.utils import get_class
from transformers import PreTrainedTokenizerBase
from transformers import pipeline as hf_transformers_pipeline

from lightning_transformers.core.config import OptimizerConfig, SchedulerConfig
from lightning_transformers.core.instantiator import Instantiator
from lightning_transformers.core.model import TaskTransformer
from lightning_transformers.core.nlp.config import HFBackboneConfig

if TYPE_CHECKING:
    from transformers import AutoModel, Pipeline


class HFTransformer(TaskTransformer):
    """Base class for task specific transformers, wrapping pre-trained language models for downstream tasks. The
    API is built on top of AutoModel and AutoConfig, provided by HuggingFace.

    see: https://huggingface.co/transformers/model_doc/auto.html

    Args:
        downstream_model_type: The AutoModel downstream model type.
            See https://huggingface.co/transformers/model_doc/auto.html
        backbone: Config containing backbone specific arguments.
        optimizer: Config containing optimizer specific arguments.
        scheduler: Config containing scheduler specific arguments.
        instantiator: Used to instantiate objects (when using Hydra).
            If Hydra is not being used the instantiator is not required,
            and functions that use instantiation such as ``configure_optimizers`` has been overridden.
        tokenizer: The pre-trained tokenizer.
        pipeline_kwargs: Arguments required for the HuggingFace inference pipeline class.
        **model_data_kwargs: Arguments passed from the data module to the class.
    """

    def __init__(
        self,
        downstream_model_type: str,
        backbone: HFBackboneConfig,
        optimizer: OptimizerConfig = OptimizerConfig(),
        scheduler: SchedulerConfig = SchedulerConfig(),
        instantiator: Optional[Instantiator] = None,
        tokenizer: Optional[PreTrainedTokenizerBase] = None,
        pipeline_kwargs: Optional[dict] = None,
        **model_data_kwargs,
    ) -> None:
        self.save_hyperparameters()
        model_cls: Type["AutoModel"] = get_class(downstream_model_type)
        model = model_cls.from_pretrained(backbone.pretrained_model_name_or_path, **model_data_kwargs)
        super().__init__(model=model, optimizer=optimizer, scheduler=scheduler, instantiator=instantiator)
        self._tokenizer = tokenizer  # necessary for hf_pipeline
        self._hf_pipeline = None
        self._hf_pipeline_kwargs = pipeline_kwargs or {}

    @property
    def tokenizer(self) -> Optional["PreTrainedTokenizerBase"]:
        if (
            self._tokenizer is None
            and hasattr(self, "trainer")  # noqa: W503
            and hasattr(self.trainer, "datamodule")  # noqa: W503
            and hasattr(self.trainer.datamodule, "tokenizer")  # noqa: W503
        ):
            self._tokenizer = self.trainer.datamodule.tokenizer
        return self._tokenizer

    @tokenizer.setter
    def tokenizer(self, tokenizer: "PreTrainedTokenizerBase") -> None:
        self._tokenizer = tokenizer

    @property
    def hf_pipeline_task(self) -> Optional[str]:
        """Override to define what HuggingFace pipeline task to use.

        Returns: Optional string to define what pipeline task to use.
        """
        return None

    @property
    def hf_pipeline(self) -> "Pipeline":
        if self._hf_pipeline is None:
            if self.hf_pipeline_task is not None:
                self._hf_pipeline = hf_transformers_pipeline(
                    task=self.hf_pipeline_task, model=self.model, tokenizer=self.tokenizer, **self._hf_pipeline_kwargs
                )
            else:
                raise RuntimeError("No task was defined for this model. Try overriding `hf_pipeline_task`")
        return self._hf_pipeline

    @hf_pipeline.setter
    def hf_pipeline(self, pipeline: "Pipeline") -> None:
        self._hf_pipeline = pipeline

    def hf_predict(self, *args, **kwargs) -> Any:
        return self.hf_pipeline(*args, **kwargs)

    def on_load_checkpoint(self, checkpoint: Dict[str, Any]) -> None:
        if "tokenizer" in checkpoint:
            self.tokenizer = checkpoint["tokenizer"]

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
        model: HFTransformer = super().load_from_checkpoint(checkpoint_path, map_location, hparams_file, strict)
        # update model with hf_pipeline_kwargs override
        if hf_pipeline_kwargs is not None:
            model._hf_pipeline_kwargs.update(hf_pipeline_kwargs)
        return model
