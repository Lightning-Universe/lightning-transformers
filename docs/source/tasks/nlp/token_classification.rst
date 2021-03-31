.. _token_classification:

Token Classification
--------------------

The Task
^^^^^^^^
The Token classification Task is similar to text classification, except each token within the text receives a prediction.
A common use of this task is Named Entity Recognition (NER). Use this task if you require your data to be classified at the token level.

Datasets
^^^^^^^^
Currently supports the `conll <https://huggingface.co/datasets/conll2003>`_ dataset, or custom input files.

Training
^^^^^^^^

.. code-block:: bash

    python train.py +task=nlp/token_classification +dataset=nlp/token_classification/conll

Swap to GPT backbone:

.. code-block:: bash

    python train.py +task=nlp/token_classification +dataset=nlp/token_classification/conll backbone.pretrained_model_name_or_path=gpt2

We report the Precision, Recall, Accuracy and Cross Entropy Loss for validation. To see all options available for the task, see `Find all options available for the task `here <https://github.com/PyTorchLightning/lightning-transformers/blob/master/conf/task/nlp/token_classification.yaml>`_.

.. include:: /datasets/nlp/token_classification_data.rst

Token Classification Inference Pipeline (experimental)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

By default we use the NER pipeline, which requires a an input sequence string and the number of labels.

For Hydra to correctly parse your input argument, if your input contains any special characters you must either wrap the entire call in single quotes like `'+x="my, sentence"'` or escape special characters. See `escaped characters in unquoted values <https://hydra.cc/docs/advanced/override_grammar/basic/#escaped-characters-in-unquoted-values>`_.

.. code-block:: bash

    python predict.py +task=nlp/token_classification +checkpoint_path=/path/to/model.ckpt +x="London is the capital of the United Kingdom." +model_data_args='{labels: 2}'

You can also run prediction using a default HuggingFace pre-trained model:

.. code-block:: bash

   python predict.py +task=nlp/token_classification +x="London is the capital of the United Kingdom." +model_data_args='{labels: 2}'

Or run prediction on a specified HuggingFace pre-trained model:

.. code-block:: bash

   python predict.py +task=nlp/token_classification backbone.pretrained_model_name_or_path=bert-base-cased +x="London is the capital of the United Kingdom." +model_data_args='{labels: 2}'
