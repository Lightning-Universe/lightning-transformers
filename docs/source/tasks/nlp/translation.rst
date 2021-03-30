Translation
-----------
The Translation task fine-tunes the model to translate text from one language to another. Currently supports the `WMT16 <https://huggingface.co/datasets/wmt16>`_ dataset or custom input text files.

.. code-block:: none

    Input Text (English): "The ground is black, the sky is blue and the car is red."
    Model Output (German): "Der Boden ist schwarz, der Himmel ist blau und das Auto ist rot."

To use this task, we must select a Seq2Seq Encoder/Decoder based model, such as multi-lingual T5 or BART. Conventional models like GPT/BERT will not work as they are encoder only.
In addition, we also need a tokenizer that has been created on multi-lingual text. This is true for `mt5 <https://huggingface.co/google/mt5-base>`_ and `mbart <https://huggingface.co/facebook/mbart-large-cc25>`_.

.. code-block:: bash

    python train.py +task=nlp/translation +dataset=nlp/translation/wmt16 backbone.pretrained_model_name_or_path=google/mt5-base

By default we fine-tune to translate from English to Romanian. This can be changed by specifying the source/target languages (see more in ``conf/dataset/nlp/translation/wmt16.yaml``):

.. code-block:: bash

    python train.py +task=nlp/translation +dataset=nlp/translation/wmt16 backbone.pretrained_model_name_or_path=google/mt5-base dataset.dataset_config_name=de-en dataset.src_lang=de dataset.tgt_lang=en

.. include:: /datasets/nlp/translation_data.rst

Translation Inference Pipeline (experimental)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

By default we use the translation pipeline, which requires a source text string.

For Hydra to correctly parse your input argument, if your input contains any special characters you must either wrap the entire call in single quotes like `'+x="my, sentence"'` or escape special characters. See `escaped characters in unquoted values <https://hydra.cc/docs/advanced/override_grammar/basic/#escaped-characters-in-unquoted-values>`_.

.. code-block:: bash

    python predict.py +task=nlp/translation +checkpoint_path=/path/to/model.ckpt +x="The ground is black, the sky is blue and the car is red."

You can also run prediction using a HuggingFace pre-trained model:

.. code-block:: bash

   python predict.py +task=nlp/translation backbone.pretrained_model_name_or_path="Helsinki-NLP/opus-mt-en-de" '+x="The ground is black, the sky is blue and the car is red."'
