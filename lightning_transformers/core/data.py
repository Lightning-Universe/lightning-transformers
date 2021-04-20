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
from typing import Any, Callable, Dict, Optional

import pytorch_lightning as pl
from torch.utils.data import DataLoader

from lightning_transformers.core.config import TransformerDataConfig


class TransformerDataModule(pl.LightningDataModule):

    def __init__(self, cfg: TransformerDataConfig = TransformerDataConfig()) -> None:
        super().__init__()
        self.cfg = cfg
        self.ds = None

    def train_dataloader(self) -> DataLoader:
        return DataLoader(
            self.ds["train"],
            batch_size=self.batch_size,
            num_workers=self.cfg.num_workers,
            collate_fn=self.collate_fn,
        )

    def val_dataloader(self) -> DataLoader:
        return DataLoader(
            self.ds["validation"],
            batch_size=self.batch_size,
            num_workers=self.cfg.num_workers,
            collate_fn=self.collate_fn,
        )

    def test_dataloader(self) -> Optional[DataLoader]:
        if "test" in self.ds:
            return DataLoader(
                self.ds["test"],
                batch_size=self.batch_size,
                num_workers=self.cfg.num_workers,
                collate_fn=self.collate_fn,
            )

    @property
    def batch_size(self) -> int:
        return self.cfg.batch_size

    @property
    def collate_fn(self) -> Optional[Callable]:
        return None

    @property
    def model_data_kwargs(self) -> Dict:
        """
        Override to provide the model with additional kwargs.
        This is useful to provide the number of classes/pixels to the model or any other data specific args
        Returns: Dict of args
        """
        return {}


class TokenizerDataModule(TransformerDataModule):

    def __init__(self, tokenizer: Any, cfg: TransformerDataConfig = TransformerDataConfig()) -> None:
        super().__init__(cfg=cfg)
        self.tokenizer = tokenizer
