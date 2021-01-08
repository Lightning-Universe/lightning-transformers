import torch

from lightning_transformers.core import TaskTransformer


class QuestionAnsweringTransformer(TaskTransformer):

    def test_step(self, batch, batch_idx, dataloader_idx=0):
        outputs = self.model(**batch)
        logits = outputs[0]
        preds = torch.argmax(logits, axis=1)
        metric_dict = self.calculate_metrics(batch, preds)
        self.log_dict(metric_dict, prog_bar=True, on_step=False, on_epoch=True)

    def calculate_metrics(self, batch, preds):
        return {}

    def create_metrics(self):
        pass

    def training_step(self, batch, batch_idx):
        outputs = self.model(**batch)
        loss = outputs[0]
        self.log('train_loss', loss)
        return loss

    def validation_step(self, batch, batch_idx, dataloader_idx=0):
        outputs = self.model(**batch)
        self.log('val_loss', outputs[0], prog_bar=True)
        return {'start_logits': outputs.start_logits, "end_logits": outputs.end_logits}

    def validation_epoch_end(self, outputs):
        # wip
        if False:
            start_logits = []
            end_logits = []
            for o in outputs:
                s_logits = o["start_logits"]
                e_logits = o["end_logits"]
                if self.trainer.use_ddp:
                    s_logits = self.all_gather(s_logits)
                    e_logits = self.all_gather(e_logits)
                start_logits.append(s_logits)
                end_logits.append(e_logits)
            start_logits = torch.cat(start_logits, dim=0)
            end_logits = torch.cat(end_logits, dim=0)
            self._calculate_metrics((start_logits, end_logits))

    def test_step(self, batch, batch_idx, dataloader_idx=0):
        outputs = self.model(**batch)
        self.log('val_loss', outputs[0], prog_bar=True)
        return {'start_logits': outputs.start_logits, "end_logits": outputs.end_logits}
