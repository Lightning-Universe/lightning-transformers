Token Classification
--------------------
Fine-tune Transformers using the Token Classification Task. Currently supports the `conll <https://huggingface.co/datasets/conll2003>`_ dataset, or custom input files.

Token classification is similar to text classification, except each token within the text receives a prediction.
A common use of this task is Named Entity Recognition (NER). Use this task if you require your data to be classified at the token level.

.. code-block:: bash

    python train.py +task=nlp/huggingface/token_classification +dataset=nlp/token_classification/conll

Swap to GPT backbone:

.. code-block:: bash

    python train.py +task=nlp/huggingface/token_classification +dataset=nlp/token_classification/conll backbone.pretrained_model_name_or_path=gpt2

We report the Precision, Recall, Accuracy and Cross Entropy Loss for validation. To see all options available for the task, see ``conf/task/nlp/huggingface/token_classification.yaml``.

Token Classification Using Custom Files
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

First we must first re-use tokens from NER as label IDs:

.. code-block:: none

    O: Outside of named entity
    B-MIS: Beginning of miscellaneous entity right after another miscellaneous entity
    I-MIS: Miscellaneous entity
    B-PER: Beginning of person’s name right after another person’s name
    I-PER: Person’s name
    B-ORG: Beginning of an organization right after another organization
    I-ORG: Organisation
    B-LOC: Beginning of a location right after another location
    I-LOC: Location

To use custom text files, the files should contain new line delimited json objects within the text files, in the same format as the `conll dataset <https://huggingface.co/datasets/conll2003#data-instances>`_:

.. code-block:: json

    {
        "chunk_tags": [11, 12, 12, 21, 13, 11, 11, 21, 13, 11, 12, 13, 11, 21, 22, 11, 12, 17, 11, 21, 17, 11, 12, 12, 21, 22, 22, 13, 11, 0],
        "id": "0",
        "ner_tags": [0, 3, 4, 0, 0, 0, 0, 0, 0, 7, 0, 0, 0, 0, 0, 7, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        "pos_tags": [12, 22, 22, 38, 15, 22, 28, 38, 15, 16, 21, 35, 24, 35, 37, 16, 21, 15, 24, 41, 15, 16, 21, 21, 20, 37, 40, 35, 21, 7],
        "tokens": ["The", "European", "Commission", "said", "on", "Thursday", "it", "disagreed", "with", "German", "advice", "to", "consumers"]
    }

.. code-block:: bash

    python train.py +task=nlp/huggingface/token_classification +dataset=nlp/token_classification/conll dataset.train_file=train.txt dataset.validation_file=valid.txt

Token Classification Inference Pipeline (under construction)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

By default we use the NER pipeline, which requires a an input sequence string.

.. code-block:: bash

    python predict.py +task=nlp/huggingface/token_classification +model=/path/to/model.ckpt +input="London is the capital of the United Kingdom."
