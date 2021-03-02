from typing import Any, Dict, Optional, TYPE_CHECKING

from hydra.utils import get_class
from transformers import pipeline as hf_transformers_pipeline

from lightning_transformers.core.config import OptimizerConfig, SchedulerConfig
from lightning_transformers.core.instantiator import Instantiator
from lightning_transformers.core.model import TaskTransformer
from lightning_transformers.core.nlp.huggingface.config import HFBackboneConfig

if TYPE_CHECKING:
    from transformers import Pipeline, PreTrainedTokenizerBase


class HFTransformer(TaskTransformer):
    """
    Base class for task specific transformers, wrapping pre-trained language models for downstream tasks.
    The API is built on top of AutoModel and AutoConfig, provided by HuggingFace.

    see: https://huggingface.co/transformers/model_doc/auto.html
    """

    def __init__(
        self,
        downstream_model_type: str,
        backbone: HFBackboneConfig,
        optimizer: OptimizerConfig,
        scheduler: SchedulerConfig,
        instantiator: Optional[Instantiator] = None,
        tokenizer: Optional["PreTrainedTokenizerBase"] = None,
        **config_data_args,
    ) -> None:
        self.save_hyperparameters()
        model = get_class(downstream_model_type).from_pretrained(
            backbone.pretrained_model_name_or_path,
            **config_data_args,
        )
        super().__init__(model=model, optimizer=optimizer, scheduler=scheduler, instantiator=instantiator)
        self.tokenizer = tokenizer  # necessary for hf_pipeline
        self.hf_pipeline = None

    @property
    def tokenizer(self) -> Optional["PreTrainedTokenizerBase"]:
        if self._tokenizer is None:
            self._tokenizer = getattr(self, "trainer.datamodule.tokenizer", None)
        return self._tokenizer

    @tokenizer.setter
    def tokenizer(self, tokenizer: "PreTrainedTokenizerBase") -> None:
        self._tokenizer = tokenizer

    @property
    def hf_pipeline_task(self) -> Optional[str]:
        """
        Override to define what HuggingFace pipeline task to use.
        Returns: Optional string to define what pipeline task to use.
        """
        return None

    @property
    def hf_pipeline(self) -> 'Pipeline':
        if self._hf_pipeline is None:
            if self.hf_pipeline_task is not None:
                self._hf_pipeline = hf_transformers_pipeline(
                    task=self.hf_pipeline_task, model=self.model, tokenizer=self.tokenizer
                )
            else:
                raise RuntimeError("No task was defined for this model. Try overriding `hf_pipeline_task`")
        return self._hf_pipeline

    @hf_pipeline.setter
    def hf_pipeline(self, pipeline: 'Pipeline') -> None:
        self._hf_pipeline = pipeline

    def hf_predict(self, *args, **kwargs) -> Any:
        return self.hf_pipeline(*args, **kwargs)

    def on_load_checkpoint(self, checkpoint: Dict[str, Any]) -> None:
        if "tokenizer" in checkpoint:
            self.tokenizer = checkpoint["tokenizer"]
