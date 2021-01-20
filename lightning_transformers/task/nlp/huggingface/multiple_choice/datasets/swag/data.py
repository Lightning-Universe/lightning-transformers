from lightning_transformers.task.nlp.huggingface.multiple_choice.core.data import MultipleChoiceTransformerDataModule


class SwagMultipleChoiceTransformerDataModule(MultipleChoiceTransformerDataModule):
    @property
    def ending_names(self) -> list:
        return [f"ending{i}" for i in range(4)]

    @property
    def context_name(self) -> str:
        return "sent1"

    @property
    def question_header_name(self) -> str:
        return "sent2"
