import argparse
import os

import pytorch_lightning as pl
from transformers import AutoTokenizer

from lightning_transformers.data import SquadDataModule
from lightning_transformers.models import LitQuestionAnsweringTransformer

# TODO is this even needed? We can pass use_fast to the tokenizer
os.environ["TOKENIZERS_PARALLELISM"] = "true"

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser = pl.Trainer.add_argparse_args(parser)
    parser = SquadDataModule.add_argparse_args(parser)
    parser = LitQuestionAnsweringTransformer.add_argparse_args(parser)
    args = parser.parse_args()
    args.do_train = False

    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path, use_fast=args.use_fast)

    dm = SquadDataModule(args, args.dataset_name, args.train_file, args.validation_file, tokenizer)
    dm.setup()

    model = LitQuestionAnsweringTransformer(args.model_name_or_path, dm.tokenizer)

    trainer = pl.Trainer.from_argparse_args(args)
    if args.do_train:
        trainer.fit(model, dm)
    trainer.test(model, datamodule=dm)
    model.save_pretrained("outputs")
