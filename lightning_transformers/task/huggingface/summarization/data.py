from lightning_transformers.core.huggingface.seq2seq.data import Seq2SeqDataModule


class SummarizationDataModule(Seq2SeqDataModule):
    @property
    def task(self) -> str:
        return "summarization"
