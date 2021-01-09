from functools import partial

from datasets import Dataset
from transformers import (
    default_data_collator,
)

from lightning_transformers.core import TransformerDataModule
from .data_utils import (
    DataCollatorForMultipleChoice,
)


class MultipleChoiceTransformerDataModule(TransformerDataModule):

    def __init__(self,
                 max_seq_length: int,
                 pad_to_max_length: int,
                 *args,
                 **kwargs):
        super().__init__(*args, **kwargs)
        self.max_seq_length = max_seq_length
        self.pad_to_max_length = pad_to_max_length

    def process_data(self, dataset: Dataset) -> Dataset:
        from .data_utils import (
            preprocess_function,
        )

        kwargs = self._prepare_preprocess_function_kwargs()
        preprocess_function = partial(preprocess_function, **kwargs)

        dataset = dataset.map(
            preprocess_function,
            batched=True,
            num_proc=self.preprocessing_num_workers,
            load_from_cache_file=self.load_from_cache_file,
        )

        return dataset

    def _prepare_preprocess_function_kwargs(self):
        kwargs = {
            "tokenizer": self.tokenizer,
            "context_name": self.context_name,
            "question_header_name": self.question_header_name,
            "ending_names": self.ending_names,
            "max_seq_length": self.max_seq_length,
            "pad_to_max_length": self.pad_to_max_length,
        }
        return kwargs

    @property
    def data_collator(self):
        return default_data_collator if self.pad_to_max_length \
            else DataCollatorForMultipleChoice(tokenizer=self.tokenizer)

    @property
    def ending_names(self) -> list:
        raise NotImplementedError

    @property
    def context_name(self) -> str:
        raise NotImplementedError

    @property
    def question_header_name(self) -> str:
        raise NotImplementedError

    @property
    def num_classes(self) -> int:
        return len(self.ending_names)

    @property
    def config_data_args(self):
        return {
            'num_labels': self.num_classes
        }
