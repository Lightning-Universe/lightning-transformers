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


def instantiate_model(model_config,
                      optimizer_config,
                      scheduler_config,
                      **kwargs):
    model = hydra.utils.instantiate(
        config=model_config,
        optim=optimizer_config,
        scheduler=scheduler_config,
        **kwargs
    )
    # model.calculate_metrics = data_module.calculate_metrics  # TODO remove this patching, put this into the model
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


def is_overridden(method_name: str, model, super_object=None) -> bool:
    assert super_object is not None
    if not hasattr(model, method_name) or not hasattr(super_object, method_name):
        # in case of calling deprecated method
        return False

    instance_attr = getattr(model, method_name)
    if not instance_attr:
        return False
    super_attr = getattr(super_object, method_name)

    # when code pointers are different, it was implemented
    if hasattr(instance_attr, 'patch_loader_code'):
        # cannot pickle __code__ so cannot verify if PatchDataloader
        # exists which shows dataloader methods have been overwritten.
        # so, we hack it by using the string representation
        is_overridden = instance_attr.patch_loader_code != str(super_attr.__code__)
    else:
        is_overridden = instance_attr.__code__ is not super_attr.__code__
    return is_overridden
