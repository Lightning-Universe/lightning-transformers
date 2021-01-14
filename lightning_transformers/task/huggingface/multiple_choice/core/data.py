from dataclasses import dataclass
from functools import partial
from typing import Optional

from datasets import Dataset, Union
from tokenizers import Tokenizer
from transformers import (
    default_data_collator,
    PreTrainedTokenizer, PreTrainedTokenizerFast
)

from lightning_transformers.core.huggingface import HFTransformerDataModule
from lightning_transformers.core.huggingface.config import HFTransformerDataConfig
from .data_utils import (
    DataCollatorForMultipleChoice,
)


@dataclass
class MultipleChoiceTransformerDataConfig(HFTransformerDataConfig):
    max_seq_length: int = 128
    pad_to_max_length: bool = True


class MultipleChoiceTransformerDataModule(HFTransformerDataModule):

    def __init__(self,
                 tokenizer: Union[Tokenizer, PreTrainedTokenizer, PreTrainedTokenizerFast],
                 cfg: MultipleChoiceTransformerDataConfig):
        super().__init__(cfg, tokenizer)
        self.cfg = cfg

    def process_data(self, dataset: Dataset, stage: Optional[str] = None) -> Dataset:
        from .data_utils import (
            preprocess_function,
        )

        kwargs = self._prepare_preprocess_function_kwargs()
        preprocess_function = partial(preprocess_function, **kwargs)

        dataset = dataset.map(
            preprocess_function,
            batched=True,
            num_proc=self.cfg.preprocessing_num_workers,
            load_from_cache_file=self.cfg.load_from_cache_file,
        )

        return dataset

    def _prepare_preprocess_function_kwargs(self):
        kwargs = {
            "tokenizer": self.tokenizer,
            "context_name": self.context_name,
            "question_header_name": self.question_header_name,
            "ending_names": self.ending_names,
            "max_seq_length": self.cfg.max_seq_length,
            "pad_to_max_length": self.cfg.pad_to_max_length,
        }
        return kwargs

    @property
    def data_collator(self):
        return default_data_collator if self.cfg.pad_to_max_length \
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
