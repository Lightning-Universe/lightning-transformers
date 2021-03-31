Language Modeling using Custom Data Processing
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Below we show an example of how to override data processing logic. In this example, we add a prefix to each block of text used in the language modeling task.

This reflects the idea of passing a conditional term that is used to give the language model context. Check :ref:`Language Modeling` for more information around the task.

Ultimately to create your own custom data processing the flow is like this:

1. Extend the ``LanguageModelingDataModule`` base class, Override hooks with your own logic
2. (Optional) Keep file in the specific task directory
3. Add a hydra config object to use your new dataset

1. Extend the ``LanguageModelingDataModule`` base class
"""""""""""""""""""""""""""""""""""""""""""""""""""""""

The base data module can be used to modify this code, and follows a simple pattern. Internally the dataset is loaded via HuggingFace Datasets, which returns an `Apache Arrow Parquet <https://arrow.apache.org/docs/python/generated/pyarrow.parquet.ParquetDataset.html>`_ Dataset. This data format is easy to transform and modify using map functions, which you'll see within the class.

.. code-block:: python

    class LanguageModelingDataModule(HFTransformerDataModule):
    cfg: LanguageModelingDataConfig # The config options passed to the constructor

    def process_data(self, dataset: Dataset, stage: Optional[str] = None) -> Dataset:
        # `process_data` converting the dataset into features.
        # The dataset is pre-loaded using `load_dataset`.
        ...
        return dataset

    @staticmethod
    def tokenize_function(
        examples,
        tokenizer: Union[Tokenizer, PreTrainedTokenizer, PreTrainedTokenizerFast],
        text_column_name: str = None,
    ):
        # tokenizes the data in a specific column using the AutoTokenizer,
        # called by `process_data`
        return tokenizer(examples[text_column_name])

    @staticmethod
    def convert_to_features(examples, block_size: int = None):
        # `process_data` calls this function to convert samples in the dataset into features
        ...

    @property
    def collate_fn(self) -> Callable:
        # `Describes how to collate the samples for the batch given to the model`
        return default_data_collator



Extend ``LanguageModelingDataModule``, like this.

.. code-block:: python

    from lightning_transformers.task.nlp.language_modeling import LanguageModelingDataModule

    class MyLanguageModelingDataModule(LanguageModelingDataModule):
        ...

Make any changes you'd like to the dataset processing via the hooks.

Below we have the pseudo code version to show where most of the changes happened within the hooks:

.. code-block:: python

    from functools import partial

    from datasets import Dataset, Optional
    from transformers import PreTrainedTokenizerBase

    from lightning_transformers.core.nlp.huggingface import HFTransformerDataConfig
    from lightning_transformers.task.nlp.language_modeling import LanguageModelingDataModule


    class MyLanguageModelingDataModule(LanguageModelingDataModule):

        def __init__(self, cfg: HFTransformerDataConfig, tokenizer: PreTrainedTokenizerBase):
            super().__init__(cfg, tokenizer)
            self.tokenized_condition_term = tokenizer("This is a story: ")

        def process_data(self, dataset: Dataset, stage: Optional[str] = None) -> Dataset:
            ...
            # Pass in our additional condition term when converting to features
            convert_to_features = partial(
                self.convert_to_features,
                block_size=self.effective_block_size,
                tokenized_condition_term=self.tokenized_condition_term
            )
            ...
            return dataset

        @staticmethod
        def convert_to_features(examples, block_size: int, **kwargs):
            # Our argument is passed in via kwargs
            tokenized_condition_term = kwargs['tokenized_condition_term']

            ...
            # Add the term to the tokenized blocks of text
            result = {
                k: [tokenized_condition_term + t[i:i + block_size] for i in range(0, total_length, block_size)]
                for k, t in concatenated_examples.items()
            }
            result["labels"] = result["input_ids"].copy()
            return result


To see the full example, see ``examples/custom/dataset/language_modeling/custom_dataset.py``

2. (Optional) Keep file in the specific task directory
""""""""""""""""""""""""""""""""""""""""""""""""""""""

This makes tracking of files easier. Our example is stored in ``examples/`` however in reality we would store our DataModule in ``lightning_transformers/task/nlp/language_modeling/custom_dataset.py``.

3. Add a hydra config object to use your new dataset
""""""""""""""""""""""""""""""""""""""""""""""""""""

Finally to use the Hydra CLI and configs, we would add our own custom yaml file containing the necessary code to run using our dataset.

We create a file at ``conf/datasets/nlp/language_modeling/my_dataset.yaml`` containing the below config.

.. code-block:: yaml

    # @package dataset
    defaults:
      - nlp/hf_default # Use the defaults from the default config found at `conf/dataset/nlp/hf_default.yaml`
    _target_: lightning_transformers.task.nlp.language_modeling.my_dataset.MyLanguageModelingDataModule # path to the class we'd like to instantiate
    cfg:
      block_size: 512 # any parameters you'd like from the inherited config object.

With this in place you can now train using either HuggingFace Datasets or your own custom files.

.. code-block:: bash

    python train.py +task=nlp/language_modeling +dataset=nlp/language_modeling/my_dataset dataset.train_file=train.csv dataset.validation_file=valid.csv
