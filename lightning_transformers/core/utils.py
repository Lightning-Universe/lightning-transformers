import inspect
import os
import shutil
import subprocess

import hydra
from hydra.utils import instantiate
from pytorch_lightning.loggers import WandbLogger
from transformers import AutoTokenizer


def initialize_wandb_logger(*args, **kwargs):
    keys = [k for k in inspect.signature(WandbLogger.__init__).parameters.keys()][1:-1]
    wandb_dict = {k: kwargs.get(k) for k in keys}

    try:
        commit_sha = (
            subprocess.check_output(["git", "rev-parse", "HEAD"])
                .decode("ascii")
                .strip()
        )
    except:
        commit_sha = "n/a"

    try:
        gitdiff = subprocess.check_output(["git", "diff"]).decode()
    except:
        gitdiff = ""

    wandb_dict["config"] = {}
    wandb_dict["config"].update(kwargs["model_config"])
    wandb_dict["config"].update(kwargs["dataset_config"])
    wandb_dict["config"].update(
        {
            "run_path": os.getcwd(),
            "commit": commit_sha,
            "notes": wandb_dict.get("notes"),
        }
    )

    wandbLogger = WandbLogger(**wandb_dict)

    shutil.copyfile(
        os.path.join(os.getcwd(), ".hydra/config.yaml"),
        os.path.join(os.getcwd(), ".hydra/hydra-config.yaml"),
    )
    wandbLogger.experiment.save(os.path.join(os.getcwd(), ".hydra/hydra-config.yaml"))
    wandbLogger.experiment.save(os.path.join(os.getcwd(), ".hydra/overrides.yaml"))

    with open("change.patch", "w") as f:
        f.write(gitdiff)
    wandbLogger.experiment.save(os.path.join(os.getcwd(), "change.patch"))

    return wandbLogger


def initialize_loggers(cfg, *args, **kwargs):
    loggers = []
    if cfg.log:
        for logger in cfg.loggers.loggers:
            loggers.append(instantiate(logger, *args, **kwargs))
    return loggers


def instantiate_downstream_model(
        task_config,
        model_config,
        optimizer_config,
        scheduler_config,
        **kwargs):
    model = hydra.utils.instantiate(
        config=task_config,
        model=model_config,
        optim=optimizer_config,
        scheduler=scheduler_config,
        **kwargs
    )
    # TODO remove this patching, put this into the model. Metrics are stored in the lightning module...
    # model.calculate_metrics = data_module.calculate_metrics
    return model


def instantiate_tokenizer(cfg):
    tokenizer = AutoTokenizer.from_pretrained(
        pretrained_model_name_or_path=cfg.pretrained_model_name_or_path,
        use_fast=cfg.task.dataset.use_fast
    )
    return tokenizer


def instantiate_data_module(dataset_config, training_config, tokenizer):
    data_module = hydra.utils.instantiate(
        config=dataset_config,
        training_config=training_config,
        tokenizer=tokenizer
    )
    return data_module
