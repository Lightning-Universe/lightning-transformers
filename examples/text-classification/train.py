import argparse

import pytorch_lightning as pl
from transformers import AutoTokenizer

from lightning_transformers.base import TransformerAdamConfig
from lightning_transformers.data import LitTransformerDataModule, TextClassificationDataModule
from lightning_transformers.models import LitTextClassificationTransformer

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser = pl.Trainer.add_argparse_args(parser)
    parser = LitTransformerDataModule.add_argparse_args(parser)
    parser = TransformerAdamConfig.add_argparse_args(parser)
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

    optim_config = TransformerAdamConfig(
        learning_rate=args.learning_rate,
        adam_epsilon=args.adam_epsilon,
        weight_decay=args.weight_decay,
        warmup_steps=args.warmup_steps
    )
    model = LitTextClassificationTransformer(
        model_name_or_path=args.model_name_or_path,
        tokenizer=tokenizer,
        optim_config=optim_config,
        num_classes=dm.num_classes
    )

    trainer = pl.Trainer.from_argparse_args(args)
    trainer.fit(model, dm)
    trainer.test(datamodule=dm)
    model.save_pretrained("outputs")
