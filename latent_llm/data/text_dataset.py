from datasets import load_dataset
from torch.utils.data import Dataset
from transformers import AutoTokenizer
import os
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
        limit: int = 100000,
        block_size: int = 1024,
        cache_dir: str = CACHE_DIR,
    ):
        self.dataset = load_dataset(dataset_id, split=split, num_proc=NUM_PROC).select(
            range(limit)
        )
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.tokenizer.pad_token = self.tokenizer.eos_token
        self.block_size = block_size
        self.dataset = self.dataset.map(
            self.tokenize,
            batched=True,
            batch_size=8,
            num_proc=NUM_PROC,
            input_columns=["text"],
            cache_file_name=f"{cache_dir}/{dataset_id}_{split}_{block_size}.bin",
            load_from_cache_file=True,
        )

    def tokenize(self, texts: list[str]) -> dict:
        input_ids = self.tokenizer(
            texts,
            return_tensors="pt",
            padding="max_length",
            truncation=True,
            max_length=self.block_size,
        ).input_ids
        return {"input_ids": input_ids}

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        return torch.tensor(self.dataset[idx]["input_ids"])


if __name__ == "__main__":
    dataset = TextDataset(
        dataset_id="anothy1/fineweb-edu-cleaned-simplified",
        split="train",
        model_name="HuggingFaceTB/SmolLM2-135M",
        limit=10000,
    )
    print(len(dataset))
    print(dataset[0])
