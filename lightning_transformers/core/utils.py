import sys
from glob import glob
import os
import os.path as osp
from typing import Union
import argparse
import hydra
from hydra.experimental import compose, initialize
import shutil
from transformers import AutoTokenizer
from omegaconf import OmegaConf

from typing import Dict
import shutil
import os
import subprocess
import hydra
import inspect
from hydra.utils import instantiate
from pytorch_lightning.loggers import WandbLogger


ROOT_DIR = osp.dirname(osp.dirname(osp.dirname(__file__)))


def initialize_WandbLogger(*args, **kwargs):

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

def copytree(src, dst, symlinks=False, ignore=None):
    if not os.path.exists(dst):
        os.makedirs(dst)
    for item in os.listdir(src):
        s = os.path.join(src, item)
        d = os.path.join(dst, item)
        if os.path.isdir(s):
            copytree(s, d, symlinks, ignore)
        else:
            if not os.path.exists(d) or os.stat(s).st_mtime - os.stat(d).st_mtime > 1:
                shutil.copy2(s, d)

def cleanup(transported_files):
    for p in transported_files:
        if os.path.exists(p):
            if os.path.isfile(p): 
                os.remove(p)
            else:
                shutil.rmtree(p)

def load_configuration():
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", type=str, default=True, help="")
    parser.add_argument("--model_name_or_path", type=str, default=True, help="")
    parser.add_argument("--dataset", type=str, default=True, help="")
    opt = parser.parse_known_args()[0]
    
    path_to_config = osp.join(ROOT_DIR, "conf")
    path_to_virtual_conf = osp.join(path_to_config, "tasks", opt.task) 
    files_to_move = [d for d in os.listdir(path_to_config) if (d != 'tasks' and d != "config.yaml")]

    try:
        transported_files = [osp.join(path_to_virtual_conf, "config.yaml")]
        shutil.copyfile(osp.join(path_to_config, "config.yaml"), transported_files[0])
        for p in files_to_move:
            src = osp.join(path_to_config, p)
            dst = osp.join(path_to_virtual_conf, p)
            transported_files.append(dst)
            copytree(src, dst)

        path_to_conf = osp.join("..", "..","conf", "tasks", opt.task) 
        initialize(path_to_conf)
        overrides = sys.argv[::]
        defaults = [v.replace('--', '') for v in overrides[1:7]]
        hydra_defaults = [f"{defaults[0]}={defaults[1]}", f"{defaults[2]}={defaults[3]}", f"{defaults[4]}={defaults[5]}"]
        cfg = compose("config.yaml", overrides=hydra_defaults + overrides[7:])
        OmegaConf.set_struct(cfg, False)
        cleanup(transported_files)
        for i in range(2, len(defaults)):
            if i % 2 == 0:
                setattr(cfg, "overrides_" + defaults[i], defaults[i + 1])
        return cfg
    except KeyboardInterrupt:
        cleanup(transported_files)
    finally:
        cleanup(transported_files)

def instantiate_model(cfg, data_module):
    model_opt = cfg.model
    model = hydra.utils.instantiate(model_opt, optim=cfg.optim)
    return model

def instantiate_data_module(cfg):
    dataset_opt = cfg.dataset
    tokenizer = AutoTokenizer.from_pretrained(cfg.overrides_model, use_fast=dataset_opt.use_fast)
    data_module = hydra.utils.instantiate(cfg.dataset, tokenizer=tokenizer, dataset_name=cfg.overrides_dataset)
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