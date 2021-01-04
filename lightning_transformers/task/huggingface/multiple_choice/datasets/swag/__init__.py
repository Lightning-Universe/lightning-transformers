from lightning_transformers.task.huggingface.multiple_choice.core.data import LitMultipleChoiceTransformerDataModule


class LitSwagMultipleChoiceTransformerDataModule(LitMultipleChoiceTransformerDataModule):

    @property
    def ending_names(self) -> list:
        return [f"ending{i}" for i in range(4)]

    @property
    def context_name(self) -> str:
        return "sent1"

    @property
    def question_header_name(self) -> str:
        return "sent2"
