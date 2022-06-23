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

from unittest.mock import MagicMock, patch

import pytest
import torch
from pytorch_lightning import LightningModule, Trainer
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader, Dataset
from transformers import AutoTokenizer

from lightning_transformers.core import TaskTransformer
from lightning_transformers.task.nlp.text_classification import (
    TextClassificationDataModule,
    TextClassificationTransformer,
)


class RandomDataset(Dataset):
    def __init__(self, size: int, length: int):
        self.len = length
        self.data = torch.randn(length, size)

    def __getitem__(self, index):
        return self.data[index]

    def __len__(self):
        return self.len


class BoringModel(LightningModule):
    def __init__(self):
        super().__init__()
        self.layer = torch.nn.Linear(32, 2)

    def forward(self, x):
        return self.layer(x)

    def loss(self, batch, prediction):
        # An arbitrary loss to have a loss that updates the model weights during `Trainer.fit` calls
        return torch.nn.functional.mse_loss(prediction, torch.ones_like(prediction))

    def step(self, x):
        x = self(x)
        out = torch.nn.functional.mse_loss(x, torch.ones_like(x))
        return out

    def training_step(self, batch, batch_idx):
        output = self(batch)
        loss = self.loss(batch, output)
        return {"loss": loss}

    def training_step_end(self, training_step_outputs):
        return training_step_outputs

    def training_epoch_end(self, outputs) -> None:
        torch.stack([x["loss"] for x in outputs]).mean()

    def validation_step(self, batch, batch_idx):
        output = self(batch)
        loss = self.loss(batch, output)
        return {"x": loss}

    def validation_epoch_end(self, outputs) -> None:
        torch.stack([x["x"] for x in outputs]).mean()

    def test_step(self, batch, batch_idx):
        output = self(batch)
        loss = self.loss(batch, output)
        return {"y": loss}

    def test_epoch_end(self, outputs) -> None:
        torch.stack([x["y"] for x in outputs]).mean()

    def configure_optimizers(self):
        optimizer = torch.optim.SGD(self.layer.parameters(), lr=0.1)
        lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1)
        return [optimizer], [lr_scheduler]

    def train_dataloader(self):
        return DataLoader(RandomDataset(32, 64))

    def val_dataloader(self):
        return DataLoader(RandomDataset(32, 64))

    def test_dataloader(self):
        return DataLoader(RandomDataset(32, 64))

    def predict_dataloader(self):
        return DataLoader(RandomDataset(32, 64))


def test_pipeline_kwargs():
    class TestModel(TaskTransformer):
        @property
        def hf_pipeline_task(self):
            return "task_name"

    cls_mock = MagicMock()
    model = TestModel(
        cls_mock, pretrained_model_name_or_path="prajjwal1/bert-tiny", pipeline_kwargs=dict(device=0), foo="bar"
    )

    with patch("lightning_transformers.core.model.hf_transformers_pipeline") as pipeline_mock:
        model.hf_pipeline
        pipeline_mock.assert_called_once_with(
            task="task_name", model=cls_mock.from_pretrained.return_value, tokenizer=None, device=0
        )


def test_task_transformer_default_optimizer_scheduler(hf_cache_path):
    tokenizer = AutoTokenizer.from_pretrained(pretrained_model_name_or_path="prajjwal1/bert-tiny")
    dm = TextClassificationDataModule(
        batch_size=1,
        dataset_name="glue",
        dataset_config_name="sst2",
        max_length=512,
        limit_test_samples=64,
        limit_val_samples=64,
        limit_train_samples=64,
        cache_dir=hf_cache_path,
        tokenizer=tokenizer,
    )
    model = TextClassificationTransformer(pretrained_model_name_or_path="prajjwal1/bert-tiny")

    trainer = Trainer(fast_dev_run=True, limit_val_batches=0, limit_test_batches=0)
    with pytest.warns(UserWarning, match="You haven't specified an optimizer or lr scheduler."):
        trainer.fit(model, dm)
    x = model.configure_optimizers()
    assert isinstance(x["optimizer"], torch.optim.AdamW)
    assert isinstance(x["lr_scheduler"]["scheduler"], LambdaLR)
