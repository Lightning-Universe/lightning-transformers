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

    python train.py +task=nlp/huggingface/language_modeling +dataset=nlp/language_modeling/wikitext

Swap to GPT backbone:

.. code-block:: bash

    python train.py +task=nlp/huggingface/language_modeling +dataset=nlp/language_modeling/wikitext backbone.pretrained_model_name_or_path=gpt2

We report the Cross Entropy Loss for validation. To see all options available for the task, see ``conf/task/nlp/huggingface/language_modeling.yaml``.

Language Modeling Using Custom Files
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

To use custom text files, the files should contain the raw data you want to train and validate on. During data pre-processing the text is flattened, and the model
is trained/validated on context windows (block size) made from the files. We override the dataset files, allowing us to still use the data transforms defined with this dataset.

.. code-block:: bash

    python train.py +task=nlp/huggingface/language_modeling +dataset=nlp/language_modeling/wikitext dataset.train_file=train.txt dataset.validation_file=valid.txt

Language Modeling Inference Pipeline (under construction)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

By default we use the text generation pipeline, which requires a conditional input string and generates an output string.

.. code-block:: bash

    python predict.py +task=nlp/huggingface/language_modeling +model=/path/to/model.ckpt +input="Condition sentence for the language model"
