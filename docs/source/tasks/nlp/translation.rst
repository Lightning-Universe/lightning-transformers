Translation
-----------
The Translation task fine-tunes the model to translate text from one language to another. Currently supports the `WMT16 <https://huggingface.co/datasets/wmt16>`_ dataset or custom input text files.

.. code-block:: none

    Input Text: <TODO>
    Model Output: <TODO>

To use this task, we must select a Seq2Seq Encoder/Decoder based model, such as multi-lingual T5 or BART. Conventional models like GPT/BERT will not work as they are encoder only.
In addition, we also need a tokenizer that has been created on multi-lingual text. This is true for `mt5 <https://huggingface.co/google/mt5-base>`_ and `mbart <https://huggingface.co/facebook/mbart-large-cc25>`_.

.. code-block:: bash

    python train.py +task=nlp/translation +dataset=nlp/translation/wmt16 backbone.pretrained_model_name_or_path=google/mt5-base

By default we fine-tune to translate from English to Romanian. This can be changed by specifying the source/target languages (see more in ``conf/dataset/nlp/translation/wmt16.yaml``):

.. code-block:: bash

    python train.py +task=nlp/translation +dataset=nlp/translation/wmt16 backbone.pretrained_model_name_or_path=google/mt5-base dataset.dataset_config_name=de-en dataset.source_language=de dataset.target_language=en

.. include:: /datasets/nlp/translation_data.rst

Translation Inference Pipeline (under construction)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

By default we use the translation pipeline, which requires a source text string.

.. code-block:: bash

    python predict.py +task=nlp/translation +model=/path/to/model.ckpt input="The ground is black, the sky is blue and the car is red."
