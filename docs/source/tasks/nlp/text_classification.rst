Text Classification
-------------------
Fine-tune Transformers using the Text Classification Task. Currently supports the `XLNI <https://huggingface.co/datasets/xlni>`_, `GLUE <https://huggingface.co/datasets/glue>`_ and `emotion <https://huggingface.co/datasets/emotion>`_ datasets, or custom input files.

Text Classification fine-tunes the model to predict probabilities across a set of labels given input text.

.. code-block:: none

    Input: I don't like this at all!

    Model answer: {"label": "angry", "score": 0.8}

Use this task when you would like to fine-tune Transformers on a labeled text classification task.
For this task, you can rely on most Transformer models as your backbone.

.. code-block:: bash

    python train.py +task=nlp/huggingface/text_classification +dataset=nlp/text_classification/emotion # can be swapped to xlni or glue

Swap to GPT backbone:

.. code-block:: bash

    python train.py +task=nlp/huggingface/text_classification +dataset=nlp/text_classification/emotion backbone.pretrained_model_name_or_path=gpt2

We report the Precision, Recall, Accuracy and Cross Entropy Loss for validation. To see all options available for the task, see ``conf/task/nlp/huggingface/text_classification.yaml``.

Text Classification Using Custom Files
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

To use custom text files, the files should contain new line delimited json objects within the text files, in the same format as  `emotion dataset <https://huggingface.co/datasets/emotion#data-instances>`_:

.. code-block:: json

    {
        "label": 0,
        "text": "im feeling quite sad and sorry for myself but ill snap out of it soon"
    }

.. code-block:: bash

    python train.py +task=nlp/huggingface/text_classification +dataset=nlp/text_classification/emotion dataset.train_file=train.txt dataset.validation_file=valid.txt

Text Classification Inference Pipeline (under construction)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

By default we use the sentiment-analysis pipeline, which requires an input string.

.. code-block:: bash

    python predict.py +task=nlp/huggingface/text_classification +model=/path/to/model.ckpt +input="I don't like this at all!"
