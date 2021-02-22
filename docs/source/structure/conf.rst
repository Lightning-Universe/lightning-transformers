.. _conf:

The Config Structure
====================

Lightning Transformers relies on Hydra config composition, meaning it is very easy to swap/modify components for training without having to touch the code.

In practice, this means you can easily experiment and try different training procedures, parameters, optimizers/schedulers extremely quickly.

Below is a diagram of the config structure, and how it fits together through the command line interface.

.. code-block:: bash

    conf/
    ┣ backbone/  # Configs defining the backbone of the model/pre-trained model if any
    ┣ dataset/ # Configs defining datasets
    ┣ optimizer/ # Configs for optimizers
    ┣ scheduler/ # Configs for schedulers
    ┣ task/ # Configs defining the task, and any task specific parameters
    ┣ tokenizer/ # Configs defining tokenizers, if any.
    ┣ trainer/ # Configs defining PyTorch Lightning Trainers, with different configurations
    ┣ training/ # Configs defining training specific parameters, such as batch size.
    ┗ config.yaml # The main entrypoint containing all our chosen config components


Hydra requires understanding of some syntactic sugar to use different configurations, which we explain below. For more details, we suggest looking at the `Hydra Documentation <https://hydra.cc/docs/next/advanced/override_grammar/basic>`_.

By default train.py loads all the defined defaults in the ``config.yaml`` file in the ``conf/`` directory.
Task and Dataset have been marked as required within the ``config.yaml``, thus have to be specified on the command line.

.. code-block:: bash

    python train.py +task=nlp/huggingface/text_classification +dataset=nlp/text_classification/emotion


``+task`` and ``+dataset`` allow us to append configs from the conf folder to our training config. These config files can be found in ``task/nlp/huggingface/text_classification`` and ``dataset/nlp/text_classification/emotion`` respectively.

Overriding Configs
******************
All default configs can be overridden with custom configs, or pre-built configs found in the ``conf/`` directory.

For example, we can swap to different optimizers like below.

.. code-block:: bash

    python train.py optimizer=sgd


We can also modify the parameters in the config object from the commandline.

.. code-block:: bash

    python train.py optimizer=sgd optimizer.momentum=0.99


We use Hydras' powerful `instantiation <https://hydra.cc/docs/next/patterns/instantiate_objects/overview>`_ to abstract the optimizer from the code, meaning it is very simple to plug in your own custom optimizers or schedulers.

.. code-block:: yaml

    # Contents of conf/optimizer/sgd

    _target_: torch.optim.SGD # _target_ class we'd like to instantiate our optimizer
    lr: ${training.lr}
    momentum: 0.9
    weight_decay: 0.0005


Config Inheritance
******************

Just like code, you can also inherit from configs.

For example, when defining a task, there is a default config object at ``task/nlp/huggingface/default.yaml`` that contains a set of default configurations for all tasks.
This is useful, as our task configs do not need to define these parameters and instead can import them, like below.

.. code-block:: yaml

    # Contents of conf/task/nlp/huggingface/language_modeling

    defaults:
      - nlp/huggingface/default # Import our default configurations
    _target_: lightning_transformers.task.nlp.huggingface.language_modeling.LanguageModelingTransformer
    downstream_model_type: transformers.AutoModelForCausalLM

These are just a subset of Hydras' features. We offer full Hydra support to provide a robust and flexible API, thus we suggest reading the `Hydra tutorials <https://hydra.cc/docs/next/tutorials/intro>`_ for advanced use cases.
