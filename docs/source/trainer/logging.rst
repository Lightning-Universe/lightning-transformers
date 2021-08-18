.. _logging:

Logging
-------

Lightning Transformers offers pre-built logging configurations out of the box to quickly get training visualizations.

.. note::

    If the logger saves to a directory, this will be located within the hydra default save directory. The configuration for this can be found in the `hydra run config <https://github.com/PyTorchLightning/lightning-transformers/blob/master/conf/hydra/output/custom.yaml>`__ and modified on the CLI, e.g ``hydra.run.dir=./``.

TensorBoard
^^^^^^^^^^^

To enable `TensorBoard <https://pytorch-lightning.readthedocs.io/en/latest/common/loggers.html#tensorboard>`__ logging to a directory, append ``+trainer/logger=tensorboard`` to your command. This appends the `tensorboard configuration <https://github.com/PyTorchLightning/lightning-transformers/blob/master/conf/trainer/logger/tensorboard.yaml>`__.

You can also modify the save directory by passing ``trainer.logger.save_dir=my_dir/``.

.. code-block:: bash

    python train.py task=nlp/multiple_choice dataset=nlp/multiple_choice/race +trainer/logger=tensorboard trainer.logger.save_dir=my_dir/


Weights & Biases
^^^^^^^^^^^^^^^^

To enable `Weights & Biases <https://pytorch-lightning.readthedocs.io/en/latest/common/loggers.html#weights-and-biases>`__ logging to a directory, append ``+trainer/logger=wandb`` to your command. This appends the `wandb configuration <https://github.com/PyTorchLightning/lightning-transformers/blob/master/conf/trainer/logger/wandb.yaml>`__.

You can also modify the project by passing ``trainer.logger.project=my_lightning_transformers_exp``.

.. code-block:: bash

    python train.py task=nlp/multiple_choice dataset=nlp/multiple_choice/race +trainer/logger=wandb trainer.logger.project=my_lightning_transformers_exp/


TestTube
^^^^^^^^

To enable `TestTube <https://pytorch-lightning.readthedocs.io/en/latest/common/loggers.html#test-tube>`__ logging to a directory, append ``+trainer/logger=testtube`` to your command. This appends the `testtube configuration <https://github.com/PyTorchLightning/lightning-transformers/blob/master/conf/trainer/logger/testtube.yaml>`__.

You can also modify the save directory by passing ``trainer.logger.save_dir=my_dir/``.

.. code-block:: bash

    python train.py task=nlp/multiple_choice dataset=nlp/multiple_choice/race +trainer/logger=testtube trainer.logger.save_dir=my_dir/
