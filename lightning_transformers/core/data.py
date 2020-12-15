import os
from typing import Optional
from functools import partial
import pytorch_lightning as pl
from datasets import load_dataset, load_metric
from torch.utils.data import DataLoader
from transformers import AutoTokenizer
from transformers import (
    default_data_collator,
    DataCollatorWithPadding,
    EvalPrediction
)
import importlib
from argparse import Namespace
from lightning_transformers.core.utils import is_overridden
from lightning_transformers import __ROOT_DIR__

class LitTransformerDataModule(pl.LightningDataModule):
    def __init__(self,
                 task: str = None,
                 training = None,
                 dataset_name: str = None,
                 train_file: str = None,
                 validation_file: str = None,
                 tokenizer: AutoTokenizer = None,
                 padding: str = 'max_length',
                 truncation: str = 'only_first',
                 max_length: int = 128,
                 preprocessing_num_workers: int = 8,
                 load_from_cache_file: bool = True,
                 dataset_config_name: Optional[str] = None,
                 train_val_split: Optional[int] = None,
                 use_fast: bool = True,
                 **kwargs):
        super().__init__()
        self.args = Namespace(**kwargs, **training)
        self.task = task
        self.dataset_name = dataset_name
        self.train_file = train_file
        self.validation_file = validation_file
        self.dataset_config_name = dataset_config_name
        self.tokenizer = tokenizer
        self.padding = padding
        self.truncation = truncation
        self.max_length = max_length
        self.preprocessing_num_workers = preprocessing_num_workers
        self.load_from_cache_file = load_from_cache_file
        self.train_val_split = train_val_split
        self.use_fast = use_fast

    def setup(self, stage: Optional[str] = None):

        self._load_processing_module()

        self._load_dataset()

        self._split_ds()

        self._process_data()

        self._prepare_labels()

        self._load_and_prepare_metrics()

    def _load_processing_module(self):
        path_to_modellib = ".".join(["lightning_transformers", "datasets", self.task, self.dataset_name, "processing"])
        self.modellib = importlib.import_module(path_to_modellib)

    def is_overridden(self, method_name):
        apply_udf = is_overridden(method_name, self, super_object=LitTransformerDataModule)
        if apply_udf:
            udf = getattr(self, method_name)
            udf()
        return apply_udf

    def _prepare_labels(self):
        pass

    def _load_and_prepare_metrics(self):
        self.is_overridden("load_and_prepare_metrics")

    @property
    def contains_test(self):
        return 'test' in self.ds

    def _pre_process(self):
        if self.is_overridden("prepare_pre_processing_functions"):
            
            if self.args.do_train:
                self.ds["train"] = self.ds["train"].map(
                    self.prepare_train_features,
                    batched=True,
                    num_proc=self.preprocessing_num_workers,
                    remove_columns=self.ds["train"].column_names,
                    load_from_cache_file=not self.load_from_cache_file,
                )
                self.train_dataloader = self._train_dataloader

            if not self.contains_test:
                self.ds["validation"] = self.ds["validation"].map(
                    self.prepare_validation_features,
                    batched=True,
                    num_proc=self.preprocessing_num_workers,
                    remove_columns=self.ds["validation"].column_names,
                    load_from_cache_file=not self.load_from_cache_file,
                )
                if self.args.do_train and self.args.do_eval:
                    self.val_dataloader = self._val_dataloader

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

    def load_and_prepare_metrics(self):
        pass

    def load_dataset(self):
        pass

    def _load_dataset(self):
        if self.is_overridden("load_dataset"):
            # User can override this function to load their own dataset.
            pass
        else:
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

    def _train_dataloader(self):
        return DataLoader(self.ds['train'], batch_size=self.args.batch_size, num_workers=self.args.num_workers, collate_fn=self.data_collator)

    def _val_dataloader(self):
        return DataLoader(self.ds['validation'], batch_size=self.args.batch_size, num_workers=self.args.num_workers, collate_fn=self.data_collator)

    def test_dataloader(self):
        dataset = self.ds['test'] if 'test' in self.ds else self.ds['validation']
        return DataLoader(dataset, batch_size=self.args.batch_size, num_workers=self.args.num_workers, collate_fn=self.data_collator)


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

    def prepare_labels(self):
        self.labels = self.ds['train'].features['labels']

    @property
    def num_classes(self):
        return self.labels.num_classes

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

class LitQuestionAnsweringTransformerDataModule(LitTransformerDataModule):

    def load_and_prepare_metrics(self):
        self.load_metrics()

        kwargs = {
            "examples": self.ds['validation'],
            "version_2_with_negative": self.args.version_2_with_negative,
            "n_best_size": self.args.n_best_size,
            "max_answer_length": self.args.max_answer_length,
            "null_score_diff_threshold": self.args.null_score_diff_threshold,
            "output_dir": self.args.output_dir,
            "is_world_process_zero": True
        }

        post_processing_function = partial(self.modellib.post_processing_function, *kwargs)

        self.calculate_metrics = partial(self.calculate_metrics, post_processing_function=post_processing_function)

    def calculate_metrics(self, features, predictions, post_processing_function=None):
        import pdb; pdb.set_trace()
        p = post_processing_function(features, predictions)
        return self.compute_metrics(p)

    def compute_metrics(self, p: EvalPrediction):
        return self.metric.compute(predictions=p.predictions, references=p.label_ids)

    def load_metrics(self):
        current_dir = os.path.sep.join(os.path.join(__file__).split(os.path.sep)[:-1])
        self.metric = load_metric(os.path.join(current_dir, "squad_v2_local") if self.args.version_2_with_negative else "squad")

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

    def prepare_pre_processing_functions(self):
        question_column_name, context_column_name, answer_column_name = self.qa_column_names

        kwargs = {"tokenizer": self.tokenizer,
                  "pad_on_right": self.pad_on_right,
                  "question_column_name": question_column_name,
                  "context_column_name": context_column_name,
                  "answer_column_name": answer_column_name,
                  "max_seq_length": self.args.max_seq_length,
                  "doc_stride": self.args.doc_stride,
                  "pad_to_max_length": self.args.pad_to_max_length}

        self.prepare_train_features = partial(self.modellib.prepare_train_features, **kwargs)
        self.prepare_validation_features = partial(self.modellib.prepare_validation_features, **kwargs)