from dataclasses import dataclass
from functools import partial
from typing import Any, Callable, Dict, Optional

from datasets import ClassLabel, Dataset
from transformers import DataCollatorForTokenClassification, PreTrainedTokenizerBase

from lightning_transformers.core.huggingface import HFTransformerDataModule
from lightning_transformers.core.huggingface.config import HFTransformerDataConfig


@dataclass
class TokenClassificationDataConfig(HFTransformerDataConfig):
    task_name: str = "ner"
    label_all_tokens: bool = False
    pad_to_max_length: bool = False


class TokenClassificationDataModule(HFTransformerDataModule):
    cfg: TokenClassificationDataConfig

    def process_data(self, dataset: Dataset, stage: Optional[str] = None) -> Dataset:
        features, label_column_name, text_column_name = self._setup_input_fields(dataset, stage)

        self._prepare_labels(dataset, features, label_column_name)

        convert_to_features = partial(
            TokenClassificationDataModule.convert_to_features,
            tokenizer=self.tokenizer,
            pad_to_max_length=self.cfg.pad_to_max_length,
            label_all_tokens=self.cfg.label_all_tokens,
            label_to_id=self.label_to_id,
            text_column_name=text_column_name,
            label_column_name=label_column_name,
        )
        dataset = dataset.map(
            convert_to_features,
            batched=True,
            num_proc=self.cfg.preprocessing_num_workers,
            load_from_cache_file=self.cfg.load_from_cache_file,
        )
        return dataset

    def _setup_input_fields(self, dataset, stage):
        if stage == "fit":
            column_names = dataset["train"].column_names
            features = dataset["train"].features
        else:
            column_names = dataset["validation"].column_names
            features = dataset["validation"].features
        text_column_name = "tokens" if "tokens" in column_names else column_names[0]
        label_column_name = (
            f"{self.cfg.task_name}_tags" if f"{self.cfg.task_name}_tags" in column_names else column_names[1]
        )
        return features, label_column_name, text_column_name

    def _prepare_labels(self, dataset, features, label_column_name) -> Optional[Any]:
        # In the event the labels are not a `Sequence[ClassLabel]`, we will need to go through the dataset to get the
        # unique labels.
        def get_label_list(labels):
            unique_labels = set()
            for label in labels:
                unique_labels = unique_labels | set(label)
            label_list = list(unique_labels)
            label_list.sort()
            return label_list

        if isinstance(features[label_column_name].feature, ClassLabel):
            label_list = features[label_column_name].feature.names
            # No need to convert the labels since they are already ints.
            label_to_id = {i: i for i in range(len(label_list))}
        else:
            label_list = get_label_list(dataset["train"][label_column_name])
            label_to_id = {l: i for i, l in enumerate(label_list)}
        num_labels = len(label_list)
        self.label_to_id, self.num_labels = label_to_id, num_labels

    @property
    def num_classes(self) -> int:
        return self.num_labels

    @property
    def config_data_args(self) -> Dict[str, int]:
        return {"num_labels": self.num_classes}

    @staticmethod
    def convert_to_features(
        examples: Any,
        tokenizer: PreTrainedTokenizerBase,
        pad_to_max_length: bool,
        label_all_tokens: bool,
        label_to_id,
        text_column_name,
        label_column_name,
    ):
        padding = "max_length" if pad_to_max_length else False

        tokenized_inputs = tokenizer(
            examples[text_column_name],
            padding=padding,
            truncation=True,
            # We use this argument because the texts in our dataset are lists of words (with a label for each word).
            is_split_into_words=True,
        )
        labels = []
        for i, label in enumerate(examples[label_column_name]):
            word_ids = tokenized_inputs.word_ids(batch_index=i)
            previous_word_idx = None
            label_ids = []
            for word_idx in word_ids:
                # Special tokens have a word id that is None. We set the label to -100 so they are automatically
                # ignored in the loss function.
                if word_idx is None:
                    label_ids.append(-100)
                # We set the label for the first token of each word.
                elif word_idx != previous_word_idx:
                    label_ids.append(label_to_id[label[word_idx]])
                # For the other tokens in a word, we set the label to either the current label or -100, depending on
                # the label_all_tokens flag.
                else:
                    label_ids.append(label_to_id[label[word_idx]] if label_all_tokens else -100)
                previous_word_idx = word_idx

            labels.append(label_ids)
        tokenized_inputs["labels"] = labels
        return tokenized_inputs

    @property
    def collate_fn(self) -> Optional[Callable]:
        return DataCollatorForTokenClassification(self.tokenizer)
