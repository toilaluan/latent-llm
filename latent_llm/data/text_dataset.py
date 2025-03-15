from datasets import load_dataset
from torch.utils.data import Dataset
from transformers import AutoTokenizer
import os
import numpy as np
import random
from mnemonic import Mnemonic
import torch


CACHE_DIR = ".training_cache/data"
NUM_PROC = 16

if not os.path.exists(CACHE_DIR):
    os.makedirs(CACHE_DIR)


class TextDataset(Dataset):
    def __init__(
        self,
        dataset_id: str,
        split: str,
        model_name: str,
        block_size: int = 1024,
    ):
        self.dataset = load_dataset(dataset_id, split=split, num_proc=NUM_PROC)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.tokenizer.add_special_tokens({"pad_token": "<|pad|>"})
        self.block_size = block_size

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        text = self.dataset[idx]["text"]
        words = text.split()
        random.shuffle(words)
        text = " ".join(words)
        input_ids = self.tokenizer(
            text,
            return_tensors="pt",
            padding="max_length",
            truncation=True,
            max_length=self.block_size,
            add_special_tokens=True,
        ).input_ids
        return input_ids.squeeze(0)


class RandomTokenDataset(Dataset):
    def __init__(self, model_name: str, block_size: int = 1024):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.tokenizer.add_special_tokens({"pad_token": "<|pad|>"})
        self.block_size = block_size
        self.vocab_size = len(self.tokenizer)
        self.all_chars = list("abcdefghijklmnopqrstuvwxyz0123456789")
        self.ids = [self.tokenizer.vocab[c] for c in self.all_chars]

    def __len__(self):
        return 100_000_000  # Arbitrary large number to simulate an unlimited dataset

    def __getitem__(self, idx):
        # Generate random token IDs within the vocabulary range
        # Exclude special tokens by using a range from 100 to vocab_size-1
        random_text = "".join(
            random.choice(self.all_chars) for _ in range(self.block_size * 10)
        )
        random_ids = self.tokenizer(
            random_text,
            return_tensors="pt",
            padding="max_length",
            truncation=True,
            max_length=self.block_size,
        ).input_ids
        # if random.random() < 0.1:
        #     n_tokens = random.randint(1, self.block_size // 2 + 1)
        # elif random.random() < 0.5:
        #     n_tokens = random.randint(self.block_size // 2 + 1, self.block_size - 1)
        # else:
        #     n_tokens = self.block_size
        # random_ids[:, n_tokens:] = self.tokenizer.pad_token_id
        return random_ids.squeeze(0)


if __name__ == "__main__":
    dataset = TextDataset(
        dataset_id="anothy1/fineweb-edu-cleaned-simplified",
        split="train",
        model_name="HuggingFaceTB/SmolLM2-135M",
        limit=10000,
    )
    print(len(dataset))
    print(dataset[0])

    random_dataset = RandomTextDataset(
        model_name="HuggingFaceTB/SmolLM2-135M", block_size=1024
    )
    print(len(random_dataset))
    print(random_dataset[0])
