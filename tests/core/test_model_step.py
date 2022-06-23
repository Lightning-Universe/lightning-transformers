import pytest
from pytorch_lightning import Callback, Trainer
from transformers import AutoTokenizer

from lightning_transformers.task.nlp.text_classification import (
    TextClassificationDataModule,
    TextClassificationTransformer,
)


class TestCallback(Callback):
    def __init__(self, expected_steps, expected_wm_steps):
        self.expected_steps = expected_steps
        self.expected_wm_steps = expected_wm_steps

    def on_fit_start(self, trainer, transformer_model: TextClassificationTransformer) -> None:
        assert transformer_model.num_training_steps == self.expected_steps
        if isinstance(self.expected_wm_steps, int):
            training_steps, warmup_steps = transformer_model.compute_warmup(-1, self.expected_wm_steps)
            assert training_steps == self.expected_steps
            assert warmup_steps == self.expected_wm_steps
        else:
            # assume float value
            training_steps, warmup_steps = transformer_model.compute_warmup(-1, self.expected_wm_steps)
            assert training_steps == self.expected_steps
            assert warmup_steps == (self.expected_steps * self.expected_wm_steps)


@pytest.mark.parametrize(
    "max_epochs,accumulate_grad_batches,expected_wm_steps",
    [
        (2, 1, 5),
        (2, 2, 5),
        (3, 2, 5),
        (3, 2, 0.5),
    ],
)
@pytest.mark.parametrize("limit_train_batches", [10])
def test_training_and_warmup_steps(
    limit_train_batches, max_epochs, accumulate_grad_batches, expected_wm_steps, hf_cache_path
):
    tokenizer = AutoTokenizer.from_pretrained(pretrained_model_name_or_path="prajjwal1/bert-tiny")
    dm = TextClassificationDataModule(
        batch_size=1,
        dataset_name="glue",
        dataset_config_name="sst2",
        max_length=512,
        limit_test_samples=64,
        limit_val_samples=64,
        limit_train_samples=64,
        cache_dir=hf_cache_path,
        tokenizer=tokenizer,
    )
    dm.setup("fit")
    model = TextClassificationTransformer(pretrained_model_name_or_path="prajjwal1/bert-tiny")

    expected_steps = (limit_train_batches // accumulate_grad_batches) * max_epochs

    trainer = Trainer(
        callbacks=TestCallback(expected_steps=expected_steps, expected_wm_steps=expected_wm_steps),
        max_epochs=max_epochs,
        accelerator="cpu",
        devices=1,
        limit_train_batches=limit_train_batches,
        limit_test_batches=0,
        limit_val_batches=0,
        accumulate_grad_batches=accumulate_grad_batches,
    )
    trainer.fit(model, dm)
