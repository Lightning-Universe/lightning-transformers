Token Classification
--------------------
The Token classification Task is similar to text classification, except each token within the text receives a prediction.
A common use of this task is Named Entity Recognition (NER). Use this task if you require your data to be classified at the token level.
Currently supports the `conll <https://huggingface.co/datasets/conll2003>`_ dataset, or custom input files.

.. code-block:: bash

    python train.py +task=nlp/token_classification +dataset=nlp/token_classification/conll

Swap to GPT backbone:

.. code-block:: bash

    python train.py +task=nlp/token_classification +dataset=nlp/token_classification/conll backbone.pretrained_model_name_or_path=gpt2

We report the Precision, Recall, Accuracy and Cross Entropy Loss for validation. To see all options available for the task, see ``conf/task/nlp/token_classification.yaml``.

.. include:: /datasets/nlp/token_classification_data.rst

Token Classification Inference Pipeline (under construction)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

By default we use the NER pipeline, which requires a an input sequence string.

.. code-block:: bash

    python predict.py +task=nlp/token_classification +model=/path/to/model.ckpt +input="London is the capital of the United Kingdom."
