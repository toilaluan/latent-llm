import os
import torch
import tiktoken
import wandb
import argparse
from datasets import load_dataset
from torch.utils.data import DataLoader

from latent_llm.models.latent_encoder import EncoderTransformer, DecoderTransformer

# Parse command line arguments
parser = argparse.ArgumentParser(description="Train a latent language model")
parser.add_argument(
    "--n_layers", type=int, default=6, help="Number of transformer layers"
)
parser.add_argument("--n_heads", type=int, default=6, help="Number of attention heads")
parser.add_argument("--embed_dim", type=int, default=384, help="Embedding dimension")
parser.add_argument("--block_size", type=int, default=1024, help="Context size")
parser.add_argument("--mem_size", type=int, default=128, help="Memory size")
parser.add_argument(
    "--n_datapoints", type=int, default=10000, help="Number of datapoints to use"
)
parser.add_argument(
    "--dataset_id",
    type=str,
    default="anothy1/fineweb-edu-cleaned-simplified",
    help="Dataset ID",
)
parser.add_argument("--batch_size", type=int, default=8, help="Batch size")
parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate")
parser.add_argument(
    "--total_steps", type=int, default=10000, help="Total training steps"
)
parser.add_argument(
    "--generate_every", type=int, default=100, help="Generate samples every N steps"
)
parser.add_argument(
    "--log_every", type=int, default=10, help="Log metrics every N steps"
)
parser.add_argument(
    "--project_name", type=str, default="latent-llm-training", help="W&B project name"
)

args = parser.parse_args()
config = vars(args)

# Add vocab_size to config (will be set after tokenizer is loaded)
config["vocab_size"] = None

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
        generated_tokens = decoder.generate(
            mem_embeds[:1, :, :], x[:1, :1], 100, padding_token
        )[0].tolist()
        generated_text = tokenizer.decode(generated_tokens)

        print("#### Original text:")
        print(original_text[:100])
        print("#### Generated text:")
        print(generated_text[:100])

        wandb.log(
            {
                "original_text": wandb.Html(original_text[:100]),
                "generated_text": wandb.Html(generated_text[:100]),
                "step": step,
            }
        )

# Save models at the end of training
torch.save(encoder.state_dict(), os.path.join(wandb.run.dir, "encoder.pt"))
torch.save(decoder.state_dict(), os.path.join(wandb.run.dir, "decoder.pt"))

wandb.finish()
