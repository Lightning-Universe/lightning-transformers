Translation using Custom Data Processing
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Below we show how to override data processing logic.

Extend the ``TranslationDataModule`` base class
"""""""""""""""""""""""""""""""""""""""""""""""

The base data module can be used to modify this code, and follows a simple pattern. Internally the dataset is loaded via HuggingFace Datasets, which returns an `Apache Arrow Parquet <https://arrow.apache.org/docs/python/generated/pyarrow.parquet.ParquetDataset.html>`_ Dataset. This data format is easy to transform and modify using map functions, which you'll see within the class.

.. code-block:: python

    class TranslationDataModule(Seq2SeqDataModule):

        @property
        def source_target_column_names(self) -> Tuple[str, str]:
            return self.cfg.source_language, self.cfg.target_language

    ...

    class Seq2SeqDataModule(HFDataModule):

        def process_data(self, dataset: Dataset, stage: Optional[str] = None) -> Dataset:
            # `process_data` converting the dataset into features.
            # The dataset is pre-loaded using `load_dataset`.
            ...
            return dataset

        @property
        def source_target_column_names(self) -> Tuple[str, str]:
            return 'source', 'target'

        @staticmethod
        def convert_to_features(examples, block_size: int = None):
            # `process_data` calls this function to convert samples in the dataset into features
            ...

        @property
        def collate_fn(self) -> Callable:
            # `Describes how to collate the samples for the batch given to the model`
            return default_data_collator


Extend ``TranslationDataModule``, like this.

.. code-block:: python

    from lightning_transformers.task.nlp.translation import TranslationDataModule

    class MyTranslationDataModule(TranslationDataModule):
        ...

Make any changes you'd like to the dataset processing via the hooks.
