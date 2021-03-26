Question Answering
------------------
The Question Answering task requires the model to determine the start and end of a span within the given context, that answers a given question.
This allows the model to pre-condition on contextual information to determine an answer. Currently supports the `SQuAD <https://huggingface.co/datasets/squad>`_ dataset or custom input text files.

.. code-block:: none

    Context: The ground is black, the sky is blue and the car is red.
    Question: What color is the sky?

    Model answer: {"answer": "the sky is blue", "start": 21, "end": 35}

Use this task when you would like to fine-tune onto data where an answer can be extracted from context information.
Since this is an extraction task, you can rely on most Transformer models as your backbone.

.. code-block:: bash

    python train.py +task=nlp/question_answering +dataset=nlp/question_answering/squad

Swap to GPT backbone:

.. code-block:: bash

    python train.py +task=nlp/question_answering +dataset=nlp/question_answering/squad backbone.pretrained_model_name_or_path=gpt2

.. include:: /datasets/nlp/question_answering_data.rst

Question Answering Inference Pipeline
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

By default we use the question answering pipeline, which requires a context and a question as input.

.. code-block:: bash

    python predict.py +task=nlp/question_Answering +model=/path/to/model.ckpt input.context="The ground is black, the sky is blue and the car is red." input.question="What color is the sky?"
