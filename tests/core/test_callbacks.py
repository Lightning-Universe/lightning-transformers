import os
import shutil

import pytest
from pl_bolts.utils import _PL_GREATER_EQUAL_1_4_5
from pytorch_lightning import Trainer

from lightning_transformers.core import callback
from lightning_transformers.core.loggers import WABLogger
from lightning_transformers.utilities.imports import _BOLTS_AVAILABLE, _BOLTS_GREATER_EQUAL_0_5_0
from tests.core.boring_model import BoringDataModule, BoringTransformerModel

if _BOLTS_GREATER_EQUAL_0_5_0:
    from pl_bolts.utils import _TORCH_MAX_VERSION_SPARSEML as _TORCH_MAX_VERSION

    MAX_VERSION = "1.10.0"
else:
    from pl_bolts.utils import _TORCH_MAX_VERSION_1_8_1 as _TORCH_MAX_VERSION

    MAX_VERSION = "1.8.1"

epoch_range_modifier = """
- !EpochRangeModifier
   start_epoch: 0.0
   end_epoch: 2.0
"""

gm_pruning_modifier = """
- !GMPruningModifier
    params: __ALL__
    init_sparsity: 0.05
    final_sparsity: 0.8
    start_epoch: 0.0
    end_epoch: 2.0
    update_frequency: 1.0
"""

quantization_modifier = """
- !QuantizationModifier
    start_epoch: 0.0
"""

set_learning_rate_modifier = """
- !SetLearningRateModifier
    start_epoch: 1.0
    learning_rate: 0.1
"""

set_weight_decay_modifier = """
- !SetWeightDecayModifier
    start_epoch: 1.0
    weight_decay: 0.1
"""


@pytest.mark.parametrize(
    "max_epochs, num_processes, limit_train_batches, modifier",
    [
        (5, 1, 5, epoch_range_modifier),
        (6, 1, 6, gm_pruning_modifier),
        (2, 1, 1, quantization_modifier),
        (5, 1, 5, set_learning_rate_modifier),
        (6, 1, 6, set_weight_decay_modifier),
    ],
)
@pytest.mark.skipif(not _BOLTS_AVAILABLE, reason="pytorch-lightning bolts not available")
@pytest.mark.skipif(not _PL_GREATER_EQUAL_1_4_5, reason="pytorch-lightning version must be >= 1.4.5")
@pytest.mark.skipif(not _TORCH_MAX_VERSION, reason=f"pytorch version must be <= {MAX_VERSION}.")
def test_training_steps(max_epochs, num_processes, limit_train_batches, modifier):
    cwd = os.getcwd()
    output_dir = os.path.join(cwd, "MODELS")
    recipe_path = os.path.join(cwd, "recipe.yaml")

    # write yaml file
    with open(recipe_path, "w") as file_handler:
        file_handler.write(modifier)

    model = BoringTransformerModel()

    data_module = BoringDataModule()

    trainer = Trainer(
        callbacks=callback.TransformerSparseMLCallback(output_dir, recipe_path),
        max_epochs=max_epochs,
        num_processes=num_processes,
        limit_train_batches=limit_train_batches,
        logger=WABLogger(offline=True),
        gpus=0,
    )

    trainer.fit(model, data_module)

    # delete yaml file
    if os.path.exists(recipe_path):
        os.remove(recipe_path)
    # delete models directory
    if os.path.exists(output_dir):
        shutil.rmtree(output_dir)
    # delete wandb folder
    if os.path.exists(os.path.join(cwd, "wandb")):
        shutil.rmtree(os.path.join(cwd, "wandb"))
