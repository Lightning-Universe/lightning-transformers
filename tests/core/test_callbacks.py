import os
import shutil

import pytest
from pytorch_lightning import Trainer
from transformers import AutoTokenizer

from lightning_transformers.callbacks import TransformerSparseMLCallback
from lightning_transformers.core.loggers import WABLogger
from lightning_transformers.task.nlp.text_classification import (
    TextClassificationDataModule,
    TextClassificationTransformer,
)
from lightning_transformers.utilities.imports import _BOLTS_AVAILABLE

if _BOLTS_AVAILABLE:
    from pl_bolts.utils import _SPARSEML_AVAILABLE

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
@pytest.mark.skipif(not _SPARSEML_AVAILABLE, reason="SparseML is not available")
@pytest.mark.skip(reason="Test current failing.")
def test_training_steps(max_epochs, num_processes, limit_train_batches, modifier, hf_cache_path):
    cwd = os.getcwd()
    output_dir = os.path.join(cwd, "MODELS")
    recipe_path = os.path.join(cwd, "recipe.yaml")

    # write yaml file
    with open(recipe_path, "w") as file_handler:
        file_handler.write(modifier)

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
    model = TextClassificationTransformer(pretrained_model_name_or_path="prajjwal1/bert-tiny")

    trainer = Trainer(
        callbacks=TransformerSparseMLCallback(output_dir, recipe_path),
        max_epochs=max_epochs,
        num_processes=num_processes,
        limit_train_batches=limit_train_batches,
        logger=WABLogger(offline=True),
        devices=0,
    )

    trainer.fit(model, dm)

    # delete yaml file
    if os.path.exists(recipe_path):
        os.remove(recipe_path)
    # delete models directory
    if os.path.exists(output_dir):
        shutil.rmtree(output_dir)
    # delete wandb folder
    if os.path.exists(os.path.join(cwd, "wandb")):
        shutil.rmtree(os.path.join(cwd, "wandb"))
