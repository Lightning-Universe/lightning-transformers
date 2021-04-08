.. _new-task:

Customizing Tasks
=================

Below we describe how you can customize the Language Modeling Task. In our example, we add weight noise when training and freeze the backbone.
For the purpose of this example we freeze the backbone within the Task, however this is recommended to be done via a `Callback <https://pytorch-lightning.readthedocs.io/en/latest/extensions/callbacks.html>`_ as seen in the `Freeze Embeddings Callback <https://github.com/PyTorchLightning/lightning-transformers/tree/master/lightning_transformers/core/nlp/seq2seq/finetuning.py>`_.

Tasks are based of a AutoModel Transformer, which handles all the internal logic when running the forward pass
through the model, and the loss calculation for a specific task. Below are the steps to customize a task within the ``LightningModule``.

1. Inherit from Lightning Transformers Base Class
2. Add custom task logic
3. Create Hydra config

1. Inherit from Lightning Transformers Base Class
-------------------------------------------------

For our example, we inherit from the Language Modeling base class.

.. code-block:: python

    from lightning_transformers.task.nlp.language_modeling import LanguageModelingTransformer

    class MyLanguageModelingTransformer(LanguageModelingTransformer):
        ...

Typically you'd store the file within the ``lightning_transformers/task/`` directory, in the appropriate task folder.
In our example, we'd store our file in ``lightning_transformers/task/language_modeling/custom_model.py``.

2. Add Custom Task Logic
------------------------

The class follows a standard ``pl.LightningModule``, thus all hooks and logic can be overridden easily.
Below we override the training step to add our logic, as well as ``on_fit_start`` to freeze the model before training.
The ``LMHeadAutoModel`` task provides separate keys for the backbone and the fully connected layer.

.. code-block:: python

    from lightning_transformers.task.nlp.language_modeling import LanguageModelingTransformer

    class MyLanguageModelingTransformer(LanguageModelingTransformer):

        def on_fit_start(self):
            super().on_fit_start()
            # Freeze backbone
            for param in self.model.transformer.parameters():
                param.requires_grad = False

        def training_step(self, batch, batch_idx):
            loss = super().training_step(batch, batch_idx)

            # Add weight noise every training step
            with torch.no_grad():
                for param in self.model.parameters():
                    param.add_(torch.randn(param.size()) * 0.1)
            return loss


3. Create Hydra Config
----------------------

Finally to use the Hydra CLI and configs, we would add our own custom yaml file containing the necessary code to run using our task.

We create a file at ``conf/task/nlp/my_language_modeling.yaml`` containing the below config.

.. code-block:: yaml

    # @package task
    defaults:
      - nlp/default # Use the defaults from the default config found at `conf/task/nlp/default.yaml`
    _target_: examples.custom_language_modeling.model.MyLanguageModelingTransformer # path to the class we'd like to instantiate
    downstream_model_type: transformers.AutoModelForCausalLM

Hydra supports config inheritence, so we could inherit from the language modeling task directly, simplifying our config a bit:

.. code-block:: yaml

    # @package task
    defaults:
      - nlp/language_modeling # Use the defaults from the config found at `conf/task/nlp/language_modeling.yaml`
    _target_: examples.custom_language_modeling.model.MyLanguageModelingTransformer # path to the class we'd like to instantiate

With this in place you can now train using pre-made HuggingFace datasets:

.. code-block:: bash

    python train.py +task=nlp/my_language_modeling dataset=nlp/language_modeling/wikitext dataset.train_file=train.csv dataset.validation_file=valid.csv

Or with your own files:

.. code-block:: bash

    python train.py +task=nlp/my_language_modeling dataset.train_file=train.csv dataset.validation_file=valid.csv
