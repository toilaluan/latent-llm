import os
import torch
import tiktoken
import wandb
from datasets import load_dataset
from torch.utils.data import DataLoader

from latent_llm.models.latent_encoder import EncoderTransformer, DecoderTransformer

# Configuration
config = {
    "n_layers": 6,
    "n_heads": 6,
    "embed_dim": 384,
    "block_size": 1024,
    "mem_size": 128,
    "n_datapoints": 10000,
    "dataset_id": "anothy1/fineweb-edu-cleaned-simplified",
    "vocab_size": None,  # Will be set after tokenizer is loaded
    "batch_size": 8,
    "lr": 1e-4,
    "total_steps": 10000,
    "generate_every": 100,
    "log_every": 10,
    "project_name": "latent-llm-training",
}

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Initialize wandb
wandb.init(project=config["project_name"], config=config)

# Tokenizer setup
tokenizer = tiktoken.get_encoding("gpt2")
config["vocab_size"] = tokenizer.n_vocab
padding_token = tokenizer.encode("<|endoftext|>", allowed_special={"<|endoftext|>"})[0]


# Data preparation
def tokenize_function(examples):
    tokenized_ids = tokenizer.encode_batch(examples["text"])
    for i in range(len(tokenized_ids)):
        tokenized_ids[i] = tokenized_ids[i][: config["block_size"]] + [
            padding_token
        ] * max(0, config["block_size"] - len(tokenized_ids[i]))
    return {
        "input_ids": tokenized_ids,
    }


def collate_fn(batch):
    return {
        "input_ids": torch.tensor([item["input_ids"] for item in batch]),
        "text": [item["text"] for item in batch],
    }


def cycle(loader):
    while True:
        for data in loader:
            yield data


# Load and prepare dataset
dataset = load_dataset(config["dataset_id"], split="train").select(
    range(config["n_datapoints"])
)
dataset = dataset.map(tokenize_function, batched=True)
dataloader = cycle(
    DataLoader(
        dataset, batch_size=config["batch_size"], shuffle=True, collate_fn=collate_fn
    )
)

# Model initialization
encoder = EncoderTransformer(
    config["vocab_size"],
    config["embed_dim"],
    config["n_heads"],
    config["block_size"],
    config["n_layers"],
    config["mem_size"],
).to(device)

decoder = DecoderTransformer(
    config["vocab_size"],
    config["embed_dim"],
    config["n_heads"],
    config["block_size"],
    config["n_layers"],
).to(device)

optimizer = torch.optim.AdamW(
    list(encoder.parameters()) + list(decoder.parameters()), lr=config["lr"]
)

# Training loop
for step in range(config["total_steps"]):
    batch = next(dataloader)
    optimizer.zero_grad()
    x = batch["input_ids"].to(device)

    # Get memory embeddings from the full sequence
    mem_embeds = encoder(x)

    # Prepare inputs and labels for the decoder
    labels = x[:, 1:]  # Shift right for next-token prediction
    decoder_input = x[:, :-1]  # Remove last token for input

    # Pass to decoder
    logits, loss = decoder(
        decoder_input, mem_embeds, labels, ignore_index=padding_token
    )
    loss.backward()
    optimizer.step()

    # Logging
    if step % config["log_every"] == 0:
        print(f"Step {step}, Loss: {loss.item()}")
        wandb.log({"loss": loss.item(), "step": step})

    # Generation for evaluation
    if step % config["generate_every"] == 0:
        original_text = batch["text"][0]
        generated_tokens = decoder.generate(mem_embeds, x[:, :1], 100, padding_token)[
            0
        ].tolist()
        generated_text = tokenizer.decode(generated_tokens)

        print("#### Original text:")
        print(original_text)
        print("#### Generated text:")
        print(generated_text)

        wandb.log(
            {
                "original_text": wandb.Html(original_text),
                "generated_text": wandb.Html(generated_text),
                "step": step,
            }
        )

# Save models at the end of training
torch.save(encoder.state_dict(), os.path.join(wandb.run.dir, "encoder.pt"))
torch.save(decoder.state_dict(), os.path.join(wandb.run.dir, "decoder.pt"))

wandb.finish()
