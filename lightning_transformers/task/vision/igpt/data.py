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
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional, Union

import numpy as np
import torch
import torchvision.transforms as T
from sklearn.cluster import KMeans, MiniBatchKMeans
from torch.utils.data import DataLoader, random_split
from torchvision import datasets
from torchvision.transforms import ToTensor

from lightning_transformers.core import TransformerDataModule
from lightning_transformers.core.config import TransformerDataConfig

DATASETS = {
    "mnist": datasets.MNIST,
    "fmnist": datasets.FashionMNIST,
    "cifar10": datasets.CIFAR10,
}


@dataclass
class ImageGPTDataConfig(TransformerDataConfig):
    dataset: str = "cifar10"
    data_dir: Optional[Union[str, Path]] = None


def find_centroids(train_x, num_clusters=16, batch_size=1024):
    pixels = train_x.reshape(-1, train_x.shape[-1])
    if batch_size:
        kmeans = MiniBatchKMeans(n_clusters=num_clusters, random_state=0, batch_size=batch_size).fit(pixels)
    else:
        kmeans = KMeans(n_clusters=num_clusters, random_state=0).fit(pixels)
    return kmeans.cluster_centers_


class ImageGPTDataModule(TransformerDataModule):
    cfg: ImageGPTDataConfig

    def __init__(self, cfg: ImageGPTDataConfig):
        super().__init__(cfg)
        self.dataset_cls = DATASETS[cfg.dataset]
        self.centroids = None
        # call fit to setup any metadata required for the model initialization
        self.setup("fit")

    def prepare_data(self, *args: Any, **kwargs: Any) -> None:
        """
        Saves files to data_dir
        """
        self.dataset_cls(self.cfg.data_dir, train=True, download=True)
        self.dataset_cls(self.cfg.data_dir, train=False, download=True)

    def setup(self, stage: Optional[str] = None):
        self.cfg.data_dir = Path(self.cfg.data_dir)

        centroid_path = self.cfg.data_dir / f"{self.cfg.dataset}_centroids.npy"
        if not os.path.exists(centroid_path):
            self._create_centroids(centroid_path)
        self.centroids = np.load(centroid_path)

        if stage == "fit" or stage is None:
            train_transforms = self.train_transforms
            val_transforms = self.test_transforms

            dataset_train = self.dataset_cls(self.cfg.data_dir, train=True, transform=train_transforms)
            dataset_val = self.dataset_cls(self.cfg.data_dir, train=True, transform=val_transforms)

            # TODO paper uses 90/10 split for every dataset besides ImageNet (96/4)
            train_size = int(0.9 * len(dataset_train))

            # Split
            # NOTE: splitting is done twice as datasets have different transforms attributes
            self.dataset_train, _ = random_split(
                dataset_train,
                [train_size, len(dataset_train) - train_size],
                generator=torch.Generator().manual_seed(0),
            )
            _, self.dataset_val = random_split(
                dataset_val,
                [train_size, len(dataset_val) - train_size],
                generator=torch.Generator().manual_seed(0),
            )

        if stage == "test" or stage is None:
            self.dataset_test = self.dataset_cls(self.cfg.data_dir, train=False, transform=self.test_transforms)

    def _create_centroids(self, centroid_path):
        self.cfg.data_dir.mkdir(exist_ok=True)
        train_ds = self.dataset_cls(self.cfg.data_dir, train=True, download=True, transform=ToTensor())
        train_x = np.stack([x.numpy() for x, _ in train_ds])
        train_x = train_x.transpose(0, 2, 3, 1)  # put channel dimension last
        centroids = find_centroids(train_x)
        np.save(centroid_path, centroids)

    @property
    def train_transforms(self):
        if self.cfg.dataset == "cifar10":
            # When full-network fine-tuning on CIFAR-10 and CIFAR100,
            # we use the augmentation popularized by Wide Residual
            # Networks: 4 pixels are reflection padded on each side, and
            # a 32 × 32 crop is randomly sampled from the padded image or its horizontal flip
            return T.Compose(
                [
                    T.RandomCrop(32, padding=4, padding_mode="reflect"),
                    T.RandomHorizontalFlip(),
                    T.ToTensor(),
                ]
            )

        elif self.cfg.dataset == "mnist" or self.cfg.dataset == "fmnist":
            return T.ToTensor()

    @property
    def test_transforms(self):
        return T.ToTensor()

    def train_dataloader(self) -> DataLoader:
        return DataLoader(
            self.dataset_train,
            batch_size=self.batch_size,
            num_workers=self.cfg.num_workers,
            collate_fn=self.collate_fn,
        )

    def val_dataloader(self) -> DataLoader:
        return DataLoader(
            self.dataset_val,
            batch_size=self.batch_size,
            num_workers=self.cfg.num_workers,
            collate_fn=self.collate_fn,
        )

    def test_dataloader(self) -> Optional[DataLoader]:
        return DataLoader(
            self.dataset_test,
            batch_size=self.batch_size,
            num_workers=self.cfg.num_workers,
            collate_fn=self.collate_fn,
        )
