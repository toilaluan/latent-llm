from datasets import load_dataset
from torch.utils.data import Dataset
from latent_llm.get_tokenizer import get_tokenizer
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
        random_masking: bool = True,
    ):
        self.dataset = load_dataset(dataset_id, split=split, num_proc=NUM_PROC)
        self.tokenizer = get_tokenizer(model_name)
        self.block_size = block_size
        self.random_masking = random_masking
        self.get_statistics()

    def get_statistics(self):
        total_tokens = 0
        total_padding = 0
        token_counts = []
        sample_size = min(1000, len(self))

        for i in range(sample_size):
            input_ids = self.__getitem__(i)
            n_tokens = (input_ids != self.tokenizer.pad_token_id).sum().item()
            token_counts.append(n_tokens)
            total_tokens += n_tokens
            total_padding += self.block_size - n_tokens

        avg_tokens = total_tokens / sample_size
        avg_padding = total_padding / sample_size
        token_counts = torch.tensor(token_counts)

        print(f"Token distribution statistics:")
        print(f"  Average tokens: {avg_tokens:.2f}")
        print(f"  Average padding: {avg_padding:.2f}")
        print(f"  Min tokens: {token_counts.min().item()}")
        print(f"  Max tokens: {token_counts.max().item()}")
        print(f"  Median tokens: {token_counts.median().item()}")
        print(f"  Fill rate: {(avg_tokens / self.block_size) * 100:.2f}%")

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

        r = random.random()
        if r < 0.4 and self.random_masking:
            beta_a, beta_b = 2.0, 1.0  # Controls shape (favors longer sequences)
            ratio = np.random.beta(beta_a, beta_b)
            n_tokens = max(1, int(ratio * self.block_size))
            input_ids[:, n_tokens:] = self.tokenizer.pad_token_id
        return input_ids.squeeze(0)


class RandomTokenDataset(Dataset):
    def __init__(self, model_name: str, block_size: int = 1024):
        self.tokenizer = get_tokenizer(model_name)
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

        # Better approach 1: Natural distribution
        # Use a more natural distribution like beta or triangular
        beta_a, beta_b = 2.0, 1.0  # Controls shape (favors longer sequences)
        ratio = np.random.beta(beta_a, beta_b)
        n_tokens = max(1, int(ratio * self.block_size))

        # Alternative: Triangular distribution
        # n_tokens = random.triangular(1, self.block_size, self.block_size * 0.8)
        # n_tokens = max(1, int(n_tokens))

        random_ids[:, n_tokens:] = self.tokenizer.pad_token_id
        return random_ids.squeeze(0)
