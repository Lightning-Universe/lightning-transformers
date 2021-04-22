.. _text_classification:

Text Classification
-------------------

The Task
^^^^^^^^
The Text Classification Task fine-tunes the model to predict probabilities across a set of labels given input text. The task supports both binary and multi-class/multi-label classification.

Datasets
^^^^^^^^
Currently supports the `XLNI <https://huggingface.co/datasets/xlni>`_, `GLUE <https://huggingface.co/datasets/glue>`_ and `emotion <https://huggingface.co/datasets/emotion>`_ datasets, or custom input files.

.. code-block:: none

    Input: I don't like this at all!

    Model answer: {"label": "angry", "score": 0.8}

Training
^^^^^^^^
Use this task when you would like to fine-tune Transformers on a labeled text classification task.
For this task, you can rely on most Transformer models as your backbone.

.. code-block:: python

    python train.py task=nlp/text_classification dataset=nlp/text_classification/emotion # can be swapped to xlni or glue

Swap to GPT backbone:

.. code-block:: python

    python train.py task=nlp/text_classification dataset=nlp/text_classification/emotion backbone.pretrained_model_name_or_path=gpt2

We report the Precision, Recall, Accuracy and Cross Entropy Loss for validation. Find all options available for the task `here <https://github.com/PyTorchLightning/lightning-transformers/blob/master/conf/task/nlp/text_classification.yaml>`_.

.. include:: /datasets/nlp/text_classification_data.rst

Text Classification Inference Pipeline (experimental)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

By default we use the sentiment-analysis pipeline, which requires an input string.

For Hydra to correctly parse your input argument, if your input contains any special characters you must either wrap the entire call in single quotes like `'+x="my, sentence"'` or escape special characters. See `escaped characters in unquoted values <https://hydra.cc/docs/advanced/override_grammar/basic/#escaped-characters-in-unquoted-values>`_.

.. code-block:: python

    python predict.py task=nlp/text_classification +checkpoint_path=/path/to/model.ckpt '+x="I dont like this at all!"'

You can also run prediction using a default HuggingFace pre-trained model:

.. code-block:: python

   python predict.py task=nlp/text_classification '+x="I dont like this at all!"'

Or run prediction on a specified HuggingFace pre-trained model:

.. code-block:: python

   python predict.py task=nlp/text_classification backbone.pretrained_model_name_or_path=bert-base-cased '+x="I dont like this at all!"'
