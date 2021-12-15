import pytest
from pytorch_lightning import Callback, Trainer

from tests.core.boring_model import BoringDataModule, BoringTransformerModel


class TestCallback(Callback):
    def __init__(self, total_steps, input_wm_steps):
        self.total_steps = total_steps
        self.input_wm_steps = input_wm_steps

    def on_fit_start(self, trainer, transformer_model: BoringTransformerModel) -> None:
        assert transformer_model.num_training_steps == self.total_steps
        if isinstance(self.input_wm_steps, int):
            training_steps, warmup_steps = transformer_model.compute_warmup(-1, self.input_wm_steps)
            assert training_steps == self.total_steps
            assert warmup_steps == self.input_wm_steps
        else:
            # assume float value
            training_steps, warmup_steps = transformer_model.compute_warmup(-1, self.input_wm_steps)
            assert training_steps == self.total_steps
            assert warmup_steps == (self.total_steps * self.input_wm_steps)
        raise SystemExit


@pytest.mark.parametrize(
    "max_epochs,num_processes,limit_train_batches,accumulate_grad_batches,input_wm_steps",
    [
        (5, 2, 5, 1, 5),
        (5, 1, 1.0, 1, 5),
        (5, 2, 1.0, 1, 5),
        (5, 1, 5, 1, 5),
        (6, 2, 6, 1, 5),
        (5, 1, 5, 2, 5),
        (6, 2, 6, 2, 5),
        (6, 2, 6, 2, 0.5),
    ],
)
def test_training_and_warmup_steps(
    max_epochs, num_processes, limit_train_batches, accumulate_grad_batches, input_wm_steps
):
    model = BoringTransformerModel()

    module = BoringDataModule()
    num_steps = len(module.train_dataloader())

    if isinstance(limit_train_batches, int) or limit_train_batches == 0.0:
        num_steps = min(num_steps, int(limit_train_batches))
    elif limit_train_batches != float("inf"):
        num_steps = int(num_steps * limit_train_batches)
    effective_batch_size = accumulate_grad_batches * num_processes
    total_steps = (num_steps // effective_batch_size) * max_epochs

    trainer = Trainer(
        callbacks=TestCallback(total_steps=total_steps, input_wm_steps=input_wm_steps),
        max_epochs=max_epochs,
        num_processes=num_processes,
        limit_train_batches=limit_train_batches,
        accumulate_grad_batches=accumulate_grad_batches,
    )
    with pytest.raises(SystemExit):
        trainer.fit(model, module)
