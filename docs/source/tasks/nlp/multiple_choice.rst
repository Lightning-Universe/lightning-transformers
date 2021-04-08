.. _multiple_choice:

Multiple Choice
---------------

The Task
^^^^^^^^
The Multiple Choice task requires the model to decide on a set of options, given a question with optional context.

Similar to the text classification task, the model is fine-tuned on multi-class classification to provide probabilities across all possible answers.
This is useful if the data you'd like the model to predict on requires selecting from a set of answers based on context or questions, where the answers can be variable.
In contrast, use the text classification task if the answers remain static and are not needed to be included during training.

Datasets
^^^^^^^^
Currently supports the `RACE <https://huggingface.co/datasets/race>`_ and `SWAG <https://huggingface.co/datasets/swag>`_ datasets, or custom input files.

.. code-block:: none

    Question: What color is the sky?
    Answers:
        A: Blue
        B: Green
        C: Red

    Model answer: A

Training
^^^^^^^^

.. code-block:: bash

    python train.py +task=nlp/multiple_choice dataset=nlp/multiple_choice/race # can use swag instead

Swap to GPT backbone:

.. code-block:: bash

    python train.py +task=nlp/multiple_choice dataset=nlp/multiple_choice/race backbone.pretrained_model_name_or_path=gpt2

We report Cross Entropy Loss, Precision, Recall and Accuracy for validation. Find all options available for the task `here <https://github.com/PyTorchLightning/lightning-transformers/blob/master/conf/task/nlp/multiple_choice.yaml>`_.

.. include:: /datasets/nlp/multiple_choice_data.rst

Multiple Choice Inference
^^^^^^^^^^^^^^^^^^^^^^^^^

Currently there is no HF pipeline available for this model. Feel free to make an issue or PR if you require this functionality.
