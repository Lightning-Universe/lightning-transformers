Multiple Choice
---------------
The Multiple Choice task requires the model to decide on a set of options, given a question with optional context. Currently supports the `RACE <https://huggingface.co/datasets/race>`_ and `SWAG <https://huggingface.co/datasets/swag>`_ datasets, or custom input files.

.. code-block:: none

    Question: What color is the sky?
    Answers:
        A: Blue
        B: Green
        C: Red

    Model answer: A

Similar to the text classification task, the model is fine-tuned on multi-class classification to provide probabilities across all possible answers.
This is useful if the data you'd like the model to predict on requires selecting from a set of answers based on context or questions, where the answers can be variable.
In contrast, use the text classification task if the answers remain static and are not needed to be included during training.

.. code-block:: bash

    python train.py +task=nlp/huggingface/multiple_choice +dataset=nlp/multiple_choice/race # can use swag instead

Swap to GPT backbone:

.. code-block:: bash

    python train.py +task=nlp/huggingface/multiple_choice +dataset=nlp/multiple_choice/race backbone.pretrained_model_name_or_path=gpt2

We report Cross Entropy Loss, Precision, Recall and Accuracy for validation. To see all options available for the task, see ``conf/task/nlp/huggingface/multiple_choice.yaml``.

Multiple Choice Using Custom Files (under construction)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

To use custom text files, the files should contain the raw data you want to train and validate on.
We override the dataset files, allowing us to still use the data transforms defined with this dataset.

.. code-block:: bash

    python train.py +task=nlp/huggingface/multiple_choice +dataset=language_modeling/race dataset.train_file=train.txt dataset.validation_file=valid.txt

Multiple Choice Inference
^^^^^^^^^^^^^^^^^^^^^^^^^

Currently there is no HF pipeline available for this model. Feel free to make an issue or PR if you require this functionality.
