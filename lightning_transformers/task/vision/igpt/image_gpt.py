# Copyright [2021] [Teddy Koker]
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
# Originally from: https://github.com/teddykoker/image-gpt
from typing import Any, Optional

import torch
import torch.nn as nn

from lightning_transformers.core import TaskTransformer
from lightning_transformers.core.config import OptimizerConfig, SchedulerConfig
from lightning_transformers.core.instantiator import Instantiator
from lightning_transformers.task.vision.igpt.data import ImageGPTDataModule
from lightning_transformers.task.vision.igpt.utils import quantize


def _to_sequence(x):
    """shape batch of images for input into GPT2 model"""
    x = x.view(x.shape[0], -1)  # flatten images into sequences
    x = x.transpose(0, 1).contiguous()  # to shape [seq len, batch]
    return x


class GenerativePixelsTransformer(TaskTransformer):
    def __init__(
        self,
        num_pixels: int,
        backbone: Any,
        optimizer: OptimizerConfig,
        scheduler: SchedulerConfig,
        instantiator: Optional[Instantiator] = None,
        classify: bool = False,
    ):
        backbone = instantiator.instantiate(backbone, num_positions=num_pixels * num_pixels)
        super().__init__(backbone, optimizer, scheduler, instantiator)
        self.save_hyperparameters()
        self.classify = classify

        self.criterion = nn.CrossEntropyLoss()

    def on_fit_start(self) -> None:
        datamodule: ImageGPTDataModule = self.trainer.datamodule
        self.centroids = nn.Parameter(torch.from_numpy(datamodule.centroids), requires_grad=False)

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        x, y = batch

        x = quantize(x, self.centroids)
        x = _to_sequence(x)

        if self.classify:
            clf_logits, logits = self.model(x, classify=True)
            clf_loss = self.criterion(clf_logits, y)
            gen_loss = self.criterion(logits.view(-1, logits.size(-1)), x.view(-1))
            # joint loss for classification
            loss = clf_loss + gen_loss
        else:
            logits = self.model(x)
            loss = self.criterion(logits.view(-1, logits.size(-1)), x.view(-1))

        logs = {"loss": loss}
        return {"loss": loss, "log": logs}

    def validation_step(self, batch, batch_idx):
        x, y = batch

        x = quantize(x, self.centroids)
        x = _to_sequence(x)

        if self.classify:
            clf_logits, logits = self.model(x, classify=True)
            clf_loss = self.criterion(clf_logits, y)
            gen_loss = self.criterion(logits.view(-1, logits.size(-1)), x.view(-1))
            # joint loss for classification
            loss = clf_loss + gen_loss
            _, preds = torch.max(clf_logits, 1)
            correct = preds == y
            return {"val_loss": loss, "correct": correct}
        else:
            logits = self.model(x)
            loss = self.criterion(logits.view(-1, logits.size(-1)), x.view(-1))
            return {"val_loss": loss}

    def validation_epoch_end(self, outs):
        avg_loss = torch.stack([x["val_loss"] for x in outs]).mean()
        logs = {"val_loss": avg_loss}
        if self.classify:
            correct = torch.cat([x["correct"] for x in outs])
            logs["val_acc"] = correct.sum().float() / correct.shape[0]
        return {"val_loss": avg_loss, "log": logs}

    def test_step(self, batch, batch_idx):
        return self.validation_step(batch, batch_idx)

    def test_epoch_end(self, outs):
        result = self.validation_epoch_end(outs)

        # replace valid stats with test stats because we are reusing function
        result["log"]["test_loss"] = result["log"].pop("val_loss")
        result["test_loss"] = result.pop("val_loss")
        if self.classify:
            result["log"]["test_acc"] = result["log"].pop("val_acc")
        return result
