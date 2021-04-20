.. _question_answering:

Question Answering
------------------

The Task
^^^^^^^^
The Question Answering task requires the model to determine the start and end of a span within the given context, that answers a given question.
This allows the model to pre-condition on contextual information to determine an answer.

Use this task when you would like to fine-tune onto data where an answer can be extracted from context information.
Since this is an extraction task, you can rely on most Transformer models as your backbone.

Datasets
^^^^^^^^
Currently supports the `SQuAD <https://huggingface.co/datasets/squad>`_ dataset or custom input text files.

.. code-block:: none

    Context: The ground is black, the sky is blue and the car is red.
    Question: What color is the sky?

    Model answer: {"answer": "the sky is blue", "start": 21, "end": 35}

Training
^^^^^^^^

.. code-block:: python

    python train.py task=nlp/question_answering dataset=nlp/question_answering/squad

Swap to GPT backbone:

.. code-block:: python

    python train.py task=nlp/question_answering dataset=nlp/question_answering/squad backbone.pretrained_model_name_or_path=gpt2

.. include:: /datasets/nlp/question_answering_data.rst

Question Answering Inference Pipeline (experimental)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

By default we use the question answering pipeline, which requires a context and a question as input.

For Hydra to correctly parse your input argument, if your input contains any special characters you must either wrap the entire call in single quotes like `'+x="my, sentence"'` or escape special characters. See `escaped characters in unquoted values <https://hydra.cc/docs/advanced/override_grammar/basic/#escaped-characters-in-unquoted-values>`_.

.. code-block:: python

    python predict.py task=nlp/question_answering +checkpoint_path=/path/to/model.ckpt +x='{context: "The ground is black, the sky is blue and the car is red.", question: "What color is the sky?"}'

You can also run prediction using a default HuggingFace pre-trained model:

.. code-block:: python

   python predict.py task=nlp/question_answering +x='{context: "The ground is black, the sky is blue and the car is red.", question: "What color is the sky?"}'

Or run prediction on a specified HuggingFace pre-trained model:

.. code-block:: python

   python predict.py task=nlp/question_answering backbone.pretrained_model_name_or_path=bert-base-cased +x='{context: "The ground is black, the sky is blue and the car is red.", question: "What color is the sky?"}'
