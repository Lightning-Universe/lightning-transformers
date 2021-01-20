from typing import Any, Dict, Optional

from hydra.utils import get_class
from pytorch_lightning import _logger as log
from transformers import pipeline

from lightning_transformers.core.config import OptimizerConfig
from lightning_transformers.core.hydra_model import HydraTaskTransformer
from lightning_transformers.core.nlp.huggingface.config import HFBackboneConfig, HFSchedulerConfig


class HFTransformer(HydraTaskTransformer):
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
        scheduler: HFSchedulerConfig,
        **config_data_args,
    ):
        super().__init__(optimizer, scheduler)
        model = get_class(downstream_model_type).from_pretrained(
            backbone.pretrained_model_name_or_path, **config_data_args
        )
        self.model = model
        self.pipeline = None
        self._tokenizer = None

    def prepare_warmup(self, cfg: HFSchedulerConfig):
        if cfg.num_training_steps < 0:
            # less than 0 specifies to infer number of training steps
            cfg.num_training_steps = self.num_training_steps
            log.info(f"Inferring number of training steps, set to {cfg.num_training_steps}")

        if isinstance(cfg.num_warmup_steps, float):
            # Convert float values to percentage of training steps to use as warmup
            cfg.num_warmup_steps *= cfg.num_training_steps
            log.info(f"Inferring number of warmup steps from ratio, set to {cfg.num_warmup_steps}")

    @property
    def tokenizer(self):
        if self._tokenizer:
            return self._tokenizer
        return self.trainer.datamodule.tokenizer

    @tokenizer.setter
    def tokenizer(self, tokenizer):
        self._tokenizer = tokenizer

    @property
    def default_pipeline_task(self) -> Optional[str]:
        """
        Override to define what HuggingFace pipeline task to use.
        Returns: Optional string to define what pipeline task to use.
        """
        return None

    @default_pipeline_task.setter
    def default_pipeline_task(self, pipeline: str):
        self._pipeline = pipeline

    def initialize_pipeline(self, pipeline_task: Optional[str] = None):
        pipeline_task = pipeline_task if pipeline_task else self.default_pipeline_task
        if pipeline_task:
            self.pipeline = pipeline(task=self.default_pipeline_task, model=self.model, tokenizer=self.tokenizer)
        else:
            raise NotImplementedError("Currently there is no support for using HuggingFace Pipelines with this task")

    def predict(self, *args, **kwargs) -> Any:
        if self.pipeline is None:
            self.initialize_pipeline()
        return self.pipeline(*args, **kwargs)

    def on_save_checkpoint(self, checkpoint: Dict[str, Any]):
        # Save tokenizer from datamodule for predict
        checkpoint["tokenizer"] = self.tokenizer

    def on_load_checkpoint(self, checkpoint: Dict[str, Any]) -> None:
        self.tokenizer = checkpoint["tokenizer"]
