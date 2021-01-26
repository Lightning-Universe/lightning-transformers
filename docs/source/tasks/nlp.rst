.. _nlp-tasks:

NLP Tasks
=========

Below are all of the NLP tasks that are supported. We use `HuggingFace Transformers <https://github.com/huggingface/transformers>`_ to create the model and tokenizers, and rely on
`Pytorch Lightning <https://www.pytorchlightning.ai/>`_.

Language Modeling
-----------------
Fine-tune Transformers on the Causal Language Modeling Task (e.g GPT-2/CTRL). Currently support the `wikitext2 <https://huggingface.co/datasets/wikitext>`_ dataset, or your own text files.

.. code-block:: bash

    python train.py +task=nlp/huggingface/language_modeling +dataset=language_modeling/wikitext

Swap to use GPT:

.. code-block:: bash

    python train.py +task=nlp/huggingface/language_modeling +dataset=language_modeling/wikitext backbone.pretrained_model_name_or_path=gpt2

We report the Cross Entropy Loss for validation. To see all options available for the task, see ``conf/task/nlp/huggingface/language_modeling.yaml``.

Using Custom Files
^^^^^^^^^^^^^^^^^^

To use custom text files, the files should contain the raw data you want to train and validate on. During data pre-processing the text is flattened, and the model
is trained/validated on context windows (block size) made from the files.

.. code-block:: bash

    python train.py +task=nlp/huggingface/language_modeling +dataset=language_modeling/wikitext dataset.train_file=train.txt dataset.validation_file=valid.txt

Inference (under construction)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

By default we use the text generation pipeline, which requires a conditional input string and generates an output string.

.. code-block:: bash

    python predict.py +task=nlp/huggingface/language_modeling +model=/path/to/model.ckpt +input="Condition sentence for the language model"

Multiple Choice
---------------

Question Answering
------------------

Summarization
-------------

Text Classification
-------------------

Token Classification
--------------------

Translation
-----------
