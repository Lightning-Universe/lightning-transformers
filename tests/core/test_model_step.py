import pytest
from pytorch_lightning import Callback, Trainer

from tests.core.boring_model import BoringDataModule, BoringTransformerModel


class TestCallback(Callback):
    def __init__(self, expected_steps, expected_wm_steps):
        self.expected_steps = expected_steps
        self.expected_wm_steps = expected_wm_steps

    def on_fit_start(self, trainer, transformer_model: BoringTransformerModel) -> None:
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
        (5, 1, 5),
        (5, 2, 5),
        (6, 2, 5),
        (6, 2, 0.5),
    ],
)
def test_training_and_warmup_steps(max_epochs, accumulate_grad_batches, expected_wm_steps):
    model = BoringTransformerModel()

    dm = BoringDataModule()
    num_steps = len(dm.train_dataloader())

    expected_steps = (num_steps // accumulate_grad_batches) * max_epochs

    trainer = Trainer(
        callbacks=TestCallback(expected_steps=expected_steps, expected_wm_steps=expected_wm_steps),
        max_epochs=max_epochs,
        accelerator="cpu",
        devices=1,
        accumulate_grad_batches=accumulate_grad_batches,
    )
    trainer.fit(model, dm)
