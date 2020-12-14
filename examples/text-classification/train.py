import argparse
import os

import pytorch_lightning as pl
from transformers import AutoTokenizer

from lightning_transformers.data import LitTransformerDataModule, TextClassificationDataModule
from lightning_transformers.models import LitTextClassificationTransformer

# TODO is this even needed? We can pass use_fast to the tokenizer
os.environ["TOKENIZERS_PARALLELISM"] = "true"

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser = pl.Trainer.add_argparse_args(parser)
    parser = LitTransformerDataModule.add_argparse_args(parser)
    parser = LitTextClassificationTransformer.add_argparse_args(parser)
    args = parser.parse_args()

    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path, use_fast=args.use_fast)

    dm = TextClassificationDataModule(
        dataset_name=args.dataset_name,
        train_file=args.train_file,
        validation_file=args.validation_file,
        tokenizer=tokenizer
    )
    dm.setup()

    model = LitTextClassificationTransformer(args.model_name_or_path, tokenizer)

    trainer = pl.Trainer.from_argparse_args(args.trainer)
    trainer.fit(model, dm)
    trainer.test(datamodule=dm)
    model.save_pretrained("outputs")
