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

import pytorch_lightning as pl

from lightning_transformers.core import TransformersBaseFinetuning


class FreezeEmbeddings(TransformersBaseFinetuning):
    """Freezes the embedding layers during training."""

    def __init__(self, train_bn: bool = True):
        super().__init__("", train_bn)

    def freeze_before_training(self, pl_module: pl.LightningModule) -> None:
        is_t5 = pl_module.model.config.model_type in ["t5", "mt5"]
        model = pl_module.model if is_t5 else pl_module.model.model
        self.freeze(modules=model.shared, train_bn=self.train_bn)
        for layer in (model.encoder, model.decoder):
            self.freeze(layer.embed_tokens)
            if not is_t5:
                self.freeze(layer.embed_positions)
