import os
from typing import Optional
from functools import partial
import pytorch_lightning as pl
from datasets import load_dataset, load_metric
from torch.utils.data import DataLoader
from transformers import AutoTokenizer
from transformers import (
    default_data_collator,
    DataCollatorWithPadding
)
from pytorch_lightning.utilities.model_utils import is_overridden

class LitTransformerDataModule(pl.LightningDataModule):
    def __init__(
            self,
            dataset_name: str,
            train_file: str,
            validation_file: str,
            tokenizer: AutoTokenizer,
            padding: str = 'max_length',
            truncation: str = 'only_first',
            max_length: int = 128,
            batch_size: int = 16,
            num_workers: int = 8,
            preprocessing_num_workers: int = 8,
            load_from_cache_file: bool = True,
            dataset_config_name: Optional[str] = None,
            train_val_split: Optional[int] = None,
            use_fast: bool = True):
        super().__init__()
        self.dataset_name = dataset_name
        self.train_file = train_file
        self.validation_file = validation_file
        self.dataset_config_name = dataset_config_name
        self.tokenizer = tokenizer
        self.padding = padding
        self.truncation = truncation
        self.max_length = max_length
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.preprocessing_num_workers = preprocessing_num_workers
        self.load_from_cache_file = load_from_cache_file
        self.train_val_split = train_val_split
        self.use_fast = use_fast

    def setup(self, stage: Optional[str] = None):

        self._load_dataset()

        self._split_ds()

        self._process_data()

        self._prepare_labels()

    def _prepare_labels(self):
        pass

    def _pre_process(self):
        self.prepare_pre_processing_functions()

        if self.args.do_train:
            self.ds["train"] = self.ds["train"].map(
                self.prepare_train_features,
                batched=True,
                num_proc=self.args.preprocessing_num_workers,
                remove_columns=self.column_names,
                load_from_cache_file=not self.args.overwrite_cache,
            )

        if not self.args.do_train:
            self.ds["validation"] = datasets["validation"].map(
                self.prepare_validation_features,
                batched=True,
                num_proc=self.args.preprocessing_num_workers,
                remove_columns=self.column_names,
                load_from_cache_file=not self.args.overwrite_cache,
            )

    def _post_process(self):
        pass

    def prepare_pre_processing_functions(self):
        return True

    def prepare_post_processing_functions(self):
        pass

    def _process_data(self):
        self._pre_process()
        self._post_process()

    def prepare_labels(self):
        pass

    def prepare_features(self):
        pass

    def _load_dataset(self):
        if self.dataset_name is not None:
            # Downloading and loading a dataset from the hub.
            self.ds = load_dataset(self.dataset_name, self.dataset_config_name)
        else:
            if not (self.train_file and self.validation_file):
                raise MisconfigurationException(
                    'You have not specified a dataset name'
                    'and need to specify a custom train file and validation file to the data module.'
                )
            data_files = {}
            if self.train_file is not None:
                data_files["train"] = self.train_file
            if self.validation_file is not None:
                data_files["validation"] = self.validation_file
            extension = self.train_file.split(".")[-1]
            self.ds = load_dataset(extension, data_files=data_files, field="data")

    def _split_ds(self):
        if self.train_val_split is not None:
            split = self.ds['train'].train_test_split(self.train_val_split)
            self.ds['train'] = split['train']
            self.ds['validation'] = split['test']

    def train_dataloader(self):
        return DataLoader(self.ds['train'], batch_size=self.batch_size, num_workers=self.num_workers, collate_fn=self.data_collator)

    def val_dataloader(self):
        return DataLoader(self.ds['validation'], batch_size=self.batch_size, num_workers=self.num_workers, collate_fn=self.data_collator)

    def test_dataloader(self):
        return DataLoader(self.ds['test'], batch_size=self.batch_size, num_workers=self.num_workers, collate_fn=self.data_collator)

    @staticmethod
    def add_argparse_args(parser):
        parser.add_argument("--dataset_name", type=str, default=None,
                            help="The name of the dataset to use (via the datasets library).")
        parser.add_argument("--dataset_config_name", type=str, default=None,
                            help="The configuration name of the dataset to use (via the datasets library).")
        parser.add_argument("--train_file", type=str, default=None, help="The input training data file (a text file).")
        parser.add_argument("--validation_file", type=str, help="The input training data file (a text file).")
        parser.add_argument("--load_from_cache_file", type=bool, default=True,
                            help="Load cached training and evaluation sets")
        parser.add_argument("--preprocessing_num_workers", type=str, default=None,
                            help="The number of processes to use for the preprocessing.")
        parser.add_argument("--max_seq_length", type=int, default=384,
                            help="The maximum total input sequence length after tokenization. Sequences longer"
                                 "than this will be truncated, sequences shorter will be padded.")
        parser.add_argument("--use_fast", type=bool, default=True, help="Use fast tokenization")
        return parser


class TextClassificationDataModule(LitTransformerDataModule):

    def prepare_features(self):

        input_feature_fields = [k for k, v in self.ds['train'].features.items() if k not in ['label', 'idx']]
        self.ds = TextClassificationDataModule.preprocess(  # TODO get @tchaton thoughts
            self.ds,
            self.tokenizer,
            input_feature_fields,
            self.padding,
            self.truncation,
            self.max_length
        )

        cols_to_keep = [
            x
            for x in ['input_ids', 'attention_mask', 'token_type_ids', 'labels', 'idx']
            if x in self.ds['train'].features
        ]
        self.ds.set_format("torch", columns=cols_to_keep)

    @staticmethod
    def convert_to_features(example_batch, indices, tokenizer, input_feature_fields, padding, truncation, max_length):
        # Either encode single sentence or sentence pairs
        if len(input_feature_fields) > 1:
            texts_or_text_pairs = list(
                zip(
                    example_batch[input_feature_fields[0]],
                    example_batch[input_feature_fields[1]]
                )
            )
        else:
            texts_or_text_pairs = example_batch[input_feature_fields[0]]

        # Tokenize the text/text pairs
        features = tokenizer.batch_encode_plus(
            texts_or_text_pairs, padding=padding, truncation=truncation, max_length=max_length
        )

        # idx is unique ID we can use to link predictions to original data
        features['idx'] = indices

        return features

    @staticmethod
    def preprocess(ds, tokenizer, input_feature_fields, padding='max_length', truncation='only_first', max_length=128):
        ds = ds.map(
            TextClassificationDataModule.convert_to_features,  # TODO get @tchaton thoughts
            batched=True,
            with_indices=True,
            fn_kwargs={
                'tokenizer': tokenizer,
                'input_feature_fields': input_feature_fields,
                'padding': padding,
                'truncation': truncation,
                'max_length': max_length,
            },
        )
        ds.rename_column_('label', "labels")
        return ds

class SquadDataModule(LitTransformerDataModule):
    dataset_name = 'squad'
    subset_name = None
    label2id = {}
    do_transform_labels = False
    train_val_split = None

    def create_metrics(self):
        current_dir = os.path.sep.join(os.path.join(__file__).split(os.path.sep)[:-1])
        metric = load_metric(os.path.join(current_dir, "squad_v2_local") if self.args.version_2_with_negative else "squad")

        self.val_dataloader = None

    @property
    def pad_on_right(self):
        return self.tokenizer.padding_side == "right"

    @property
    def column_names(self):
        if self.args.do_train:
            return self.ds["train"].column_names
        else:
            return self.ds["validation"].column_names

    @property
    def data_collator(self):
        return default_data_collator if self.args.pad_to_max_length else DataCollatorWithPadding(self.tokenizer)

    @property
    def qa_column_names(self):
        question_column_name = "question" if "question" in self.column_names else self.column_names[0]
        context_column_name = "context" if "context" in self.column_names else self.column_names[1]
        answer_column_name = "answers" if "answers" in self.column_names else self.column_names[2]
        return question_column_name, context_column_name, answer_column_name

    def prepare_data_processing_map_functions(self):
        from lightning_transformers.question_answering.squad_preparation import (
            prepare_train_features,
            prepare_validation_features
        )

        question_column_name, context_column_name, answer_column_name = self.qa_column_names

        kwargs = {"tokenizer": self.tokenizer,
                  "pad_on_right": self.pad_on_right,
                  "question_column_name": question_column_name,
                  "context_column_name": context_column_name,
                  "answer_column_name": answer_column_name,
                  "max_seq_length": self.args.max_seq_length,
                  "doc_stride": self.args.doc_stride,
                  "pad_to_max_length": self.args.pad_to_max_length}

        return partial(prepare_train_features, **kwargs), partial(prepare_validation_features, **kwargs)

    @staticmethod
    def add_argparse_args(parser):
        parser = LitTransformerDataModule.add_argparse_args(parser)
        parser.add_argument("--pad_to_max_length", type=bool, default=True,
                            help="Whether to pad all samples to `max_seq_length`. "
                                 "If False, will pad the samples dynamically when batching to the maximum length in the batch (which can "
                                 "be faster on GPU but will be slower on TPU).")
        parser.add_argument("--version_2_with_negative", type=bool, default=False,
                            help="If true, some of the examples do not have an answer.")
        parser.add_argument("--null_score_diff_threshold", type=float, default=.0,
                            help="The threshold used to select the null answer: if the best answer has a score that is less than "
                                 "the score of the null answer minus this threshold, the null answer is selected for this example. "
                                 "Only useful when `version_2_with_negative=True`.")
        parser.add_argument("--doc_stride", type=int, default=128,
                            help="When splitting up a long document into chunks, how much stride to take between chunks.")
        parser.add_argument("--n_best_size", type=int, default=20,
                            help="The total number of n-best predictions to generate when looking for an answer.")
        parser.add_argument("--max_answer_length", type=int, default=30,
                            help="The maximum length of an answer that can be generated. This is needed because the start "
                                 "and end predictions are not conditioned on one another.")
        return parser
