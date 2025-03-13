from datasets import load_dataset
from torch.utils.data import Dataset
from transformers import AutoTokenizer
import os
import torch
from nltk.corpus import words
import random
from mnemonic import Mnemonic


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
        ).input_ids
        n_tokens = random.randint(1, self.block_size)
        input_ids[0, :n_tokens] = self.tokenizer.pad_token_id
        return input_ids.squeeze(0)


class RandomTextDataset(Dataset):
    def __init__(self, model_name: str, block_size: int = 1024):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.tokenizer.add_special_tokens({"pad_token": "<|pad|>"})
        self.block_size = block_size
        self.mnemo = Mnemonic("english")

    def _random_long_text(self) -> str:
        return " ".join(self._random_text() for _ in range(random.randint(1, 32)))

    def _random_text(self) -> str:
        return self.mnemo.generate(strength=256)

    def __len__(self):
        return 100_000_000  # Arbitrary large number to simulate an unlimited dataset

    def __getitem__(self, idx):
        text = self._random_long_text()
        input_ids = self.tokenizer(
            text,
            return_tensors="pt",
            padding="max_length",
            truncation=True,
            max_length=self.block_size,
        ).input_ids
        n_tokens = random.randint(1, self.block_size)
        input_ids[0, :n_tokens] = self.tokenizer.pad_token_id
        return input_ids.squeeze(0)


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
