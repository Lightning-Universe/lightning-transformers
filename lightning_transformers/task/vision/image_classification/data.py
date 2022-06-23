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
from typing import Any, Optional

from datasets import ClassLabel, Dataset
from pytorch_lightning.utilities import rank_zero_warn

from lightning_transformers.core import TransformerDataModule


class ImageClassificationDataModule(TransformerDataModule):
    """Defines the ``LightningDataModule`` for Image Classification Datasets."""

    def __init__(self, feature_extractor, *args, **kwargs) -> None:
        super().__init__(tokenizer=feature_extractor, *args, **kwargs)
        self.labels = None

    def process_data(self, dataset: Dataset, stage: Optional[str] = None) -> Dataset:
        input_feature_fields = [
            k for k, v in dataset["train"].features.items() if k not in ["label", "idx", "labels", "image_file_path"]
        ]

        self.input_feature_fields = input_feature_fields
        dataset = dataset.with_transform(self.convert_to_features)

        if not isinstance(dataset["train"].features["labels"], ClassLabel):
            dataset = dataset.class_encode_column("labels")

        self.labels = dataset["train"].features["labels"]
        return dataset

    def convert_to_features(self, example_batch: Any):
        images = example_batch[self.input_feature_fields[0]]
        inputs = self.tokenizer([x for x in images], batched=True, with_indices=True, return_tensors="pt")
        inputs["labels"] = example_batch["labels"]
        return inputs

    @property
    def num_classes(self) -> int:
        if self.labels is None:
            rank_zero_warn("Labels has not been set, calling `setup('fit')`.")
            self.setup("fit")
        return self.labels.num_classes
