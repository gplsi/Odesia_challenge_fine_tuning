# Copyright Lightning AI. Licensed under the Apache License 2.0, see LICENSE file.

import json
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Set, Union
import pandas as pd

import torch
from torch.utils.data import DataLoader

from prepare_data.prompts import PromptStyle
from prepare_data.base import DataModule, SFTDataset, get_sft_collate_fn
from transformers import AutoTokenizer as Tokenizer

_URL = "https://huggingface.co/datasets/Muennighoff/flan/resolve/main"


# TODO: Including all subsets, FLAN is too large to be loaded in memory. Switch the implementation to cache
#   on disk or use Lightning Data
@dataclass
class FLAN(DataModule):
    """FLAN data module for supervised finetuning."""

    mask_prompt: bool = True
    """Whether to mask the prompt section from the label (with ``ignore_index``)."""
    prompt_style: Union[str, PromptStyle] = "flan"
    """The style to apply to instruction prompts. See `litgpt.prompts` for a list of available styles."""
    ignore_index: int = -100
    """The index to use for elements to be ignored in the label."""
    seed: int = 42
    """The random seed for shuffling the dataset."""
    num_workers: int = 4
    """How many DataLoader processes to use for loading."""
    download_dir: Path = Path("./data/flan")
    """The directory in which the downloaded dataset gets saved."""
    url: str = _URL
    """The URL from where to download the dataset."""
    subsets: Optional[str] = None
    """A comma separated list of subsets to use. If None, all subsets are used."""

    tokenizer: Optional[Tokenizer] = field(default=None, init=False, repr=False)
    train_batch_size: int = field(default=1, init=False, repr=False)
    eval_batch_size: int = field(default=1, init=False, repr=False)
    max_seq_length: int = field(default=-1, init=False, repr=False)
    train_dataset: Optional[SFTDataset] = field(default=None, init=False, repr=False)
    test_dataset: Optional[SFTDataset] = field(default=None, init=False, repr=False)

    def __post_init__(self):
        if isinstance(self.prompt_style, str):
            self.prompt_style = PromptStyle.from_name(self.prompt_style)


    def connect(
        self, tokenizer: Optional[Tokenizer] = None, num_workers: int =4, train_batch_size: int = 1, test_batch_size: int = 1, max_seq_length: Optional[int] = None,
            train_dataset: str = None, test_dataset: str = None
    ) -> None:
        self.tokenizer = tokenizer
        self.num_workers = num_workers
        self.train_batch_size = train_batch_size
        self.test_batch_size = test_batch_size
        self.max_seq_length = -1 if max_seq_length is None else max_seq_length
        self.train_dataset = train_dataset
        self.test_dataset = test_dataset

    def prepare_data(self) -> None:
        dir = os.path.join(os.getcwd(), self.train_dataset)
        list_files = os.listdir(dir)
        df_all_train = pd.DataFrame()

        for file in list_files[0:1]:
            df_temp = pd.read_excel(os.path.join(dir, file))
            df_temp.rename(columns={'inputs': 'instruction', 'targets': 'output'}, inplace=True)
            df_all_train = pd.concat([df_all_train, df_temp], axis=0)

        dir = os.path.join(os.getcwd(), self.test_dataset)
        list_files = os.listdir(dir)
        df_all_test = pd.DataFrame()
        for file in list_files[0:1]:
            df_temp = pd.read_excel(os.path.join(dir, file))
            df_temp.rename(columns={'inputs': 'instruction', 'targets': 'output'}, inplace=True)
            df_all_test = pd.concat([df_all_test, df_temp], axis=0)

        self.train_dataset = SFTDataset(
            data=df_all_train,
            tokenizer=self.tokenizer,
            prompt_style=self.prompt_style,
            max_seq_length=self.max_seq_length,
            mask_prompt=self.mask_prompt,
            ignore_index=self.ignore_index,
        )
        self.test_dataset = SFTDataset(
            data=df_all_test,
            tokenizer=self.tokenizer,
            prompt_style=self.prompt_style,
            max_seq_length=self.max_seq_length,
            mask_prompt=self.mask_prompt,
            ignore_index=self.ignore_index,
        )
        train_dataloader = DataLoader(
            self.train_dataset,
            batch_size=self.train_batch_size,
            shuffle=True,
            #generator=torch.Generator().manual_seed(self.seed),
            num_workers=self.num_workers,
            collate_fn=get_sft_collate_fn(max_seq_length=self.max_seq_length, ignore_index=self.ignore_index),
        )
        eval_dataloader = DataLoader(
            self.test_dataset,
            batch_size=self.test_batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            collate_fn=get_sft_collate_fn(max_seq_length=self.max_seq_length, ignore_index=self.ignore_index),
        )
        return train_dataloader, eval_dataloader