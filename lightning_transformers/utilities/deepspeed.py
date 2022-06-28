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
from typing import Any

import pytorch_lightning as pl
import transformers.deepspeed


class ZeRO3Config:
    def __init__(self, pl_module):
        self.config = pl_module.trainer.strategy.config

    def __call__(self, *args, **kwargs) -> Any:
        return self

    def is_zero3(self) -> bool:
        return self.config.get("zero_optimization") and self.config.get("zero_optimization").get("stage") == 3


def enable_transformers_pretrained_deepspeed_sharding(pl_module: "pl.LightningModule") -> None:
    transformers.deepspeed._hf_deepspeed_config_weak_ref = ZeRO3Config(pl_module)
