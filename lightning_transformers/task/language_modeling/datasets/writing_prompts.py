import os
import tarfile
from typing import Union, List, Optional

import hydra
import torch
import pytorch_lightning as pl
import wget
from omegaconf import DictConfig
from torch.utils.data import Dataset, DataLoader

WRITING_PROMPTS_DOWNLOAD_URL = "https://dl.fbaipublicfiles.com/fairseq/data/writingPrompts.tar.gz"


class WritingPromptsDataModule(pl.LightningDataModule):
    def __init__(self,
                 tokenizer: DictConfig,
                 file_dir: str = './',
                 download_dir: str = './',
                 prompt_end_token: str = '<EOP>',
                 seq_len: int = 128):
        super().__init__()
        self.tokenizer = hydra.utils.instantiate(tokenizer)
        self.file_dir = file_dir
        self.download_dir = download_dir
        self.prompt_end_token = prompt_end_token
        self.seq_len = seq_len

    def prepare_data(self):
        if not os.path.exists(os.path.join(self.file_dir, 'writingPrompts/')):
            file_path = wget.download(WRITING_PROMPTS_DOWNLOAD_URL, self.download_dir)
            with tarfile.open(file_path) as f:
                f.extractall(self.file_dir)

    def setup(self, stage: Optional[str] = None):
        self.val_prompts, self.val_targets = self._load('val.wp_source', 'val.wp_target')
        self.train_prompts, self.train_targets = self._load('train.wp_source', 'train.wp_target')

    def train_dataloader(self) -> DataLoader:
        dataset = WritingPromptsDataset(
            prompts=self.train_prompts,
            targets=self.train_targets,
            prompt_end_token=self.prompt_end_token,
            seq_len=self.seq_len
        )
        return DataLoader(dataset)

    def val_dataloader(self) -> Union[DataLoader, List[DataLoader]]:
        dataset = WritingPromptsDataset(
            prompts=self.val_prompts,
            targets=self.val_targets,
            prompt_end_token=self.prompt_end_token,
            seq_len=self.seq_len
        )
        return DataLoader(dataset)

    def _load(self, source_filename, target_filename):
        source_filename = os.path.join(self.file_dir, source_filename)
        target_filename = os.path.join(self.file_dir, target_filename)
        with open(source_filename) as f:
            prompts = f.readlines()
        with open(target_filename) as f:
            targets = f.readlines()
        prompts = self.tokenizer.tokenize(prompts)
        targets = self.tokenizer.tokenizer(targets)
        return prompts, targets


class WritingPromptsDataset(Dataset):
    def __init__(self, prompts, targets, prompt_end_token, seq_len):
        super().__init__()
        self.prompts = prompts
        self.targets = targets
        self.prompt_end_token = prompt_end_token
        self.seq_len = seq_len

    def __getitem__(self, index):
        prompt = self.prompts[index]
        target = self.targets[index]

        if len(target) > self.seq_len:
            rand_start = torch.randint(0, len(target) - self.seq_len - len(prompt) - 1, (1,))
            target = target[rand_start: self.seq_len + 1]

        sample = prompt + self.prompt_end_token + target
        return sample

    def __len__(self):
        return len(self.prompts)


class Tokenizer:
    # ! pip install tokenizers

    from pathlib import Path

    from tokenizers import ByteLevelBPETokenizer

    paths = [str(x) for x in Path("./eo_data/").glob("**/*.txt")]

    # Initialize a tokenizer
    tokenizer = ByteLevelBPETokenizer()

    # Customize training
    tokenizer.train(files=paths, vocab_size=52_000, min_frequency=2, special_tokens=[
        "<s>",
        "<pad>",
        "</s>",
        "<unk>",
        "<mask>",
    ])

    # Save files to disk
    tokenizer.save_model(".", "esperberto")
