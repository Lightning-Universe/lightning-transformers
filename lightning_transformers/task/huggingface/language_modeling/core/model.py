from lightning_transformers.core.model import TaskTransformer


class LanguageModelingTransformer(TaskTransformer):

    def on_fit_start(self):
        tokenizer = self.trainer.datamodule.tokenizer
        tokenizer_length = len(tokenizer)
        self.model.resize_token_embeddings(tokenizer_length)

    def _step(self, batch, batch_idx):
        outputs = self.model(**batch)
        loss, logits = outputs[:2]
        return loss

    def training_step(self, batch, batch_idx):
        loss = self._step(batch, batch_idx)
        self.log('train_loss', loss)
        return loss

    def validation_step(self, batch, batch_idx, dataloader_idx=0):
        loss = self._step(batch, batch_idx)
        self.log('val_loss', loss, sync_dist=True)

    def test_step(self, batch, batch_idx, dataloader_idx=0):
        loss = self._step(batch, batch_idx)
        self.log('test_loss', loss, sync_dist=True)
