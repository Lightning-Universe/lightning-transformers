import time
from typing import Any, Optional

import torch
from omegaconf import DictConfig, OmegaConf
from pytorch_lightning import Callback
from pytorch_lightning.utilities.distributed import rank_zero_info

from lightning_transformers.core import TaskTransformer, TransformerDataModule
from lightning_transformers.core.config import TaskConfig, TrainerConfig, TransformerDataConfig
from lightning_transformers.core.instantiator import HydraInstantiator, Instantiator
from lightning_transformers.core.nlp.config import TokenizerConfig
from lightning_transformers.core.utils import set_ignore_warnings


class CUDACallback(Callback):

    def on_train_epoch_start(self, trainer, pl_module):
        # Reset the memory use counter
        torch.cuda.reset_peak_memory_stats(trainer.root_gpu)
        torch.cuda.synchronize(trainer.root_gpu)
        self.start_time = time.time()

    def on_train_epoch_end(self, trainer, pl_module, outputs):
        torch.cuda.synchronize(trainer.root_gpu)
        max_memory = torch.cuda.max_memory_allocated(trainer.root_gpu) / 2**20
        epoch_time = time.time() - self.start_time

        max_memory = torch.tensor(max_memory, dtype=torch.int, device=trainer.root_gpu)
        epoch_time = torch.tensor(epoch_time, dtype=torch.int, device=trainer.root_gpu)

        torch.distributed.all_reduce(max_memory, op=torch.distributed.ReduceOp.SUM)
        torch.distributed.all_reduce(epoch_time, op=torch.distributed.ReduceOp.SUM)

        world_size = torch.distributed.get_world_size()

        print(f"Average Epoch time: {epoch_time.item() / float(world_size):.2f} seconds")
        print(f"Average Peak memory {max_memory.item() / float(world_size):.2f}MiB")


def run(
    instantiator: Instantiator,
    ignore_warnings: bool = True,
    do_train: bool = True,
    dataset: TransformerDataConfig = TransformerDataConfig(),
    task: TaskConfig = TaskConfig(),
    trainer: TrainerConfig = TrainerConfig(),
    tokenizer: Optional[TokenizerConfig] = None,
    logger: Optional[Any] = None,
):
    if ignore_warnings:
        set_ignore_warnings()

    data_module_kwargs = {}
    if tokenizer is not None:
        data_module_kwargs["tokenizer"] = tokenizer

    data_module: TransformerDataModule = instantiator.data_module(dataset, **data_module_kwargs)
    data_module.setup("fit")

    model: TaskTransformer = instantiator.model(task, model_data_args=data_module.model_data_args)
    trainer = instantiator.trainer(trainer, logger=logger, callbacks=[CUDACallback()])

    if do_train:
        trainer.fit(model, datamodule=data_module)
    trainer.test(model, datamodule=data_module)


def main(cfg: DictConfig):
    rank_zero_info(OmegaConf.to_yaml(cfg))
    instantiator = HydraInstantiator()
    logger = instantiator.logger(cfg)
    run(
        instantiator,
        ignore_warnings=cfg.ignore_warnings,
        do_train=cfg.training.do_train,
        dataset=cfg.dataset,
        tokenizer=cfg.tokenizer if "tokenizer" in cfg else None,
        task=cfg.task,
        trainer=cfg.trainer,
        logger=logger,
    )
