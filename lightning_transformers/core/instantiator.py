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
import logging
from typing import Optional, TYPE_CHECKING, Union

import hydra
import pytorch_lightning as pl
import torch
from omegaconf import DictConfig

from lightning_transformers.core import TransformerDataModule
from lightning_transformers.core.data import TokenizerDataModule

if TYPE_CHECKING:
    # avoid circular imports
    from lightning_transformers.core import TaskTransformer


class Instantiator:

    def model(self, *args, **kwargs):
        raise NotImplementedError("Child class must implement method")

    def optimizer(self, *args, **kwargs):
        raise NotImplementedError("Child class must implement method")

    def scheduler(self, *args, **kwargs):
        raise NotImplementedError("Child class must implement method")

    def data_module(self, *args, **kwargs):
        raise NotImplementedError("Child class must implement method")

    def logger(self, *args, **kwargs):
        raise NotImplementedError("Child class must implement method")

    def trainer(self, *args, **kwargs):
        raise NotImplementedError("Child class must implement method")

    def instantiate(self, *args, **kwargs):
        raise NotImplementedError("Child class must implement method")


class HydraInstantiator(Instantiator):

    def model(
        self,
        cfg: DictConfig,
        model_data_kwargs: Optional[DictConfig] = None,
        tokenizer: Optional[DictConfig] = None,
        pipeline_kwargs: Optional[DictConfig] = None
    ) -> "TaskTransformer":
        if model_data_kwargs is None:
            model_data_kwargs = {}
        model_data_kwargs = dict(model_data_kwargs)  # avoid ConfigKeyError: Key 'tokenizer' is not in struct`

        # use `model_data_kwargs` to pass `tokenizer` and `pipeline_kwargs`
        # as not all models might contain these parameters.
        if tokenizer:
            model_data_kwargs["tokenizer"] = self.instantiate(tokenizer)
        if pipeline_kwargs:
            model_data_kwargs["pipeline_kwargs"] = pipeline_kwargs

        return self.instantiate(cfg, instantiator=self, **model_data_kwargs)

    def optimizer(self, model: torch.nn.Module, cfg: DictConfig) -> torch.optim.Optimizer:
        no_decay = ["bias", "LayerNorm.weight"]
        grouped_parameters = [
            {
                "params": [
                    p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay) and p.requires_grad
                ],
                "weight_decay": cfg.weight_decay,
            },
            {
                "params": [
                    p for n, p in model.named_parameters() if any(nd in n for nd in no_decay) and p.requires_grad
                ],
                "weight_decay": 0.0,
            },
        ]
        return self.instantiate(cfg, grouped_parameters)

    def scheduler(self, cfg: DictConfig, optimizer: torch.optim.Optimizer) -> torch.optim.lr_scheduler._LRScheduler:
        return self.instantiate(cfg, optimizer=optimizer)

    def data_module(
        self,
        cfg: DictConfig,
        tokenizer: Optional[DictConfig] = None,
    ) -> Union[TransformerDataModule, TokenizerDataModule]:
        if tokenizer:
            return self.instantiate(cfg, tokenizer=self.instantiate(tokenizer))
        return self.instantiate(cfg)

    def logger(self, cfg: DictConfig) -> Optional[logging.Logger]:
        if cfg.get("log"):
            return self.instantiate(cfg.logger)

    def trainer(self, cfg: DictConfig, **kwargs) -> pl.Trainer:
        return self.instantiate(cfg, **kwargs)

    def instantiate(self, *args, **kwargs):
        return hydra.utils.instantiate(*args, **kwargs)
