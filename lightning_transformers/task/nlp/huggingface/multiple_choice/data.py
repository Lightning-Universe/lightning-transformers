from transformers import default_data_collator

from lightning_transformers.core.nlp.huggingface import HFTransformerDataModule
from lightning_transformers.task.nlp.huggingface.multiple_choice.utils import DataCollatorForMultipleChoice


class MultipleChoiceTransformerDataModule(HFTransformerDataModule):
    @property
    def pad_to_max_length(self):
        return self.cfg.padding == "max_length"

    @property
    def collate_fn(self):
        return (
            default_data_collator if self.pad_to_max_length else DataCollatorForMultipleChoice(tokenizer=self.tokenizer)
        )

    @property
    def num_classes(self) -> int:
        raise NotImplementedError

    @property
    def model_data_args(self):
        return {"num_labels": self.num_classes}
