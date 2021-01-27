.. _nlp-tasks:

NLP Tasks
=========

Below are all of the NLP tasks that are supported. We use `HuggingFace Transformers <https://github.com/huggingface/transformers>`_ to create the Transformer model and tokenizers, whilst relying on
`Pytorch Lightning <https://www.pytorchlightning.ai/>`_ for training.

By relying on Lightning for training we're able to use the suite of tools Lightning has to offer, such as
training optimizations, multi-node support, powerful override functionalities provided by the LightningModule and various Callbacks such as Stochastic Weight Averaging and Pruning.
Using Lightning also allows us to support custom fine-tuning strategies which can be seen in <TODO>.

Language Modeling
-----------------
Fine-tune Transformers using the Causal Language Modeling Task (e.g GPT-3/CTRL). Currently supports the `wikitext2 <https://huggingface.co/datasets/wikitext>`_ dataset, or custom input files.

Causal Language Modeling is the vanilla autoregressive pre-training method common to most language models
(Excluding BERT-like models, which were pre-trained using the Masked Language Modeling training method).
During training, we minimize the maximum likelihood during training across spans of text data (usually in some context window/block size).
When trained on large quantities of text data, this gives us strong language models such as GPT-3 to use for downstream tasks.

Since this task is usually the pre-training task for Transformers, it can be used to train new language models from scratch or to fine-tune a language model onto your own unlabeled text data.

Language Models pre-trained or fine-tuned to the Causal Language Modeling task can then be used in generative predictions, see <TODO> for more.

.. code-block:: bash

    python train.py +task=nlp/huggingface/language_modeling +dataset=nlp/language_modeling/wikitext

Swap to GPT backbone:

.. code-block:: bash

    python train.py +task=nlp/huggingface/language_modeling +dataset=nlp/language_modeling/wikitext backbone.pretrained_model_name_or_path=gpt2

We report the Cross Entropy Loss for validation. To see all options available for the task, see ``conf/task/nlp/huggingface/language_modeling.yaml``.

Using Custom Files
^^^^^^^^^^^^^^^^^^

To use custom text files, the files should contain the raw data you want to train and validate on. During data pre-processing the text is flattened, and the model
is trained/validated on context windows (block size) made from the files.

.. code-block:: bash

    python train.py +task=nlp/huggingface/language_modeling +dataset=nlp/language_modeling/wikitext dataset.train_file=train.txt dataset.validation_file=valid.txt

Inference Pipeline (under construction)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

By default we use the text generation pipeline, which requires a conditional input string and generates an output string.

.. code-block:: bash

    python predict.py +task=nlp/huggingface/language_modeling +model=/path/to/model.ckpt +input="Condition sentence for the language model"

Multiple Choice
---------------
Fine-tune Transformers using the Multiple Choice Task. Currently supports the `RACE <https://huggingface.co/datasets/race>`_ and `SWAG <https://huggingface.co/datasets/swag>`_ datasets, or custom input files.

.. code-block:: bash

    python train.py +task=nlp/huggingface/multiple_choice +dataset=nlp/multiple_choice/race # can use swag instead

Swap to GPT backbone:

.. code-block:: bash

    python train.py +task=nlp/huggingface/multiple_choice +dataset=nlp/multiple_choice/race backbone.pretrained_model_name_or_path=gpt2

We report Cross Entropy Loss, Precision, Recall and Accuracy for validation. To see all options available for the task, see ``conf/task/nlp/huggingface/multiple_choice.yaml``.

Using Custom Files (under construction)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

To use custom text files, the files should contain the raw data you want to train and validate on. During data pre-processing the text is flattened, and the model
is trained/validated on context windows (block size) made from the files.

.. code-block:: bash

    python train.py +task=nlp/huggingface/language_modeling +dataset=language_modeling/wikitext dataset.train_file=train.txt dataset.validation_file=valid.txt

Inference (under construction)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

By default we use the text generation pipeline, which requires a conditional input string and generates an output string.

.. code-block:: bash

    python predict.py +task=nlp/huggingface/language_modeling +model=/path/to/model.ckpt +input="Condition sentence for the language model"

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
