from lightning_transformers.core.nlp import HFTransformer


class LanguageModelingTransformer(HFTransformer):

    def __init__(self, *args, downstream_model_type: str = 'transformers.AutoModelForCausalLM', **kwargs) -> None:
        super().__init__(downstream_model_type, *args, **kwargs)

    def on_fit_start(self):
        tokenizer_length = len(self.tokenizer)
        self.model.resize_token_embeddings(tokenizer_length)

    def _step(self, batch, batch_idx):
        outputs = self.model(**batch)
        loss = outputs[0]
        return loss

    def training_step(self, batch, batch_idx):
        loss = self._step(batch, batch_idx)
        self.log("train_loss", loss)
        return loss

    def validation_step(self, batch, batch_idx, dataloader_idx=0):
        loss = self._step(batch, batch_idx)
        self.log("val_loss", loss, sync_dist=True)

    def test_step(self, batch, batch_idx, dataloader_idx=0):
        loss = self._step(batch, batch_idx)
        self.log("test_loss", loss, sync_dist=True)

    @property
    def hf_pipeline_task(self) -> str:
        return "text-generation"
