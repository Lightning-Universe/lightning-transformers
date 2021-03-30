Language Modeling
-----------------
Causal Language Modeling is the vanilla autoregressive pre-training method common to most language models such as GPT-3 or CTRL
(Excluding BERT-like models, which were pre-trained using the Masked Language Modeling training method).
Currently supports the `wikitext2 <https://huggingface.co/datasets/wikitext>`_ dataset, or custom input files.

During training, we minimize the maximum likelihood during training across spans of text data (usually in some context window/block size).
The model is able to attend to the left context (left of the mask).
When trained on large quantities of text data, this gives us strong language models such as GPT-3 to use for downstream tasks.

Since this task is usually the pre-training task for Transformers, it can be used to train new language models from scratch or to fine-tune a language model onto your own unlabeled text data.

Language Models pre-trained or fine-tuned to the Causal Language Modeling task can then be used in generative predictions, see <TODO> for more.

.. code-block:: bash

    python train.py +task=nlp/language_modeling +dataset=nlp/language_modeling/wikitext

Swap to GPT backbone:

.. code-block:: bash

    python train.py +task=nlp/language_modeling +dataset=nlp/language_modeling/wikitext backbone.pretrained_model_name_or_path=gpt2

We report the Cross Entropy Loss for validation. To see all options available for the task, see ``conf/task/nlp/language_modeling.yaml``.

.. include:: /datasets/nlp/language_modeling_data.rst

Language Modeling Inference Pipeline (experimental)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

By default we use the text generation pipeline, which requires a conditional input string and generates an output string.

For Hydra to correctly parse your input argument, if your input contains any special characters you must either wrap the entire call in single quotes like `'+x="my, sentence"'` or escape special characters. See `Escaped characters in unquoted values <https://hydra.cc/docs/advanced/override_grammar/basic/#escaped-characters-in-unquoted-values>`_.

.. code-block:: bash

    python predict.py +task=nlp/language_modeling +checkpoint_path=/path/to/model.ckpt +x="Condition sentence for the language model"

You can also run prediction using a default HuggingFace pre-trained model:

.. code-block:: bash

   python predict.py +task=nlp/language_modeling +x="Condition sentence for the language model"

Or run prediction on a specified HuggingFace pre-trained model:

.. code-block:: bash

   python predict.py +task=nlp/language_modeling backbone.pretrained_model_name_or_path=bert-base-cased +x="Condition sentence for the language model"
