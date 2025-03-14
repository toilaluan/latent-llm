import argparse
import torch
import os
from transformers import AutoTokenizer
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
import wandb
import random
import numpy as np
import pandas as pd
from latent_llm.models.gpt_latent import LatentEncoder, LatentDecoder
from latent_llm.models.gpt_latent_flow_matching import GPTLatentFlowMatching
from latent_llm.data.text_dataset import TextDataset

VAE_SHIFT = 0
VAE_SCALE = 1.0


class TextCompletionDataset(Dataset):
    def __init__(
        self,
        dataset_id: str,
        split: str,
        tokenizer,
        block_size: int = 1024,
        min_prefix_length: int = 10,
        max_prefix_ratio: float = 0.7,
    ):
        """
        Text completion dataset that splits text into prefix and suffix.

        Args:
            dataset_id: HuggingFace dataset ID
            split: Dataset split (train, validation, test)
            tokenizer: Tokenizer for encoding text
            block_size: Maximum sequence length
            min_prefix_length: Minimum number of tokens in prefix
            max_prefix_ratio: Maximum ratio of prefix to total text length
        """
        from datasets import load_dataset

        self.dataset = load_dataset(dataset_id, split=split)
        self.tokenizer = tokenizer
        self.block_size = block_size
        self.min_prefix_length = min_prefix_length
        self.max_prefix_ratio = max_prefix_ratio

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        # Get text from dataset
        text = self.dataset[idx]["text"]

        # Calculate approximate character length for splitting
        approx_char_count = len(text)

        # Determine split point based on character count
        max_char_prefix = int(approx_char_count * self.max_prefix_ratio)

        # Split text into prefix and suffix at character level
        if max_char_prefix < approx_char_count:
            # Random split point between min_prefix_length chars and max_ratio
            # Using character approximation for min_prefix_length (assuming ~4 chars per token)
            min_char_prefix = min(self.min_prefix_length * 4, max_char_prefix)
            if max_char_prefix > min_char_prefix:
                split_point = random.randint(min_char_prefix, max_char_prefix)
            else:
                split_point = min_char_prefix

            prefix_text = text[:split_point]
            suffix_text = text[split_point:]
        else:
            # If text is too short, use most of it as prefix
            prefix_text = text[:max_char_prefix]
            suffix_text = text[max_char_prefix:]

        # Now tokenize the prefix and suffix separately
        prefix_tokens = self.tokenizer(
            prefix_text,
            truncation=True,
            max_length=self.block_size,
            padding="max_length",
            return_tensors="pt",
        ).input_ids[0]

        suffix_tokens = self.tokenizer(
            suffix_text,
            truncation=True,
            max_length=self.block_size,
            padding="max_length",
            return_tensors="pt",
        ).input_ids[0]

        return {
            "prefix": prefix_tokens,
            "suffix": suffix_tokens,
        }


def parse_args():
    parser = argparse.ArgumentParser(
        description="Train a GPT Latent Flow Matching model for text prediction"
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default="anothy1/fineweb-edu-cleaned-simplified",
        help="Dataset to use for training",
    )
    parser.add_argument(
        "--model_name",
        type=str,
        default="HuggingFaceTB/SmolLM2-135M",
        help="Base model name to use",
    )
    parser.add_argument(
        "--encoder_id",
        type=str,
        required=True,
        help="Pretrained encoder model ID on HuggingFace Hub",
    )
    parser.add_argument(
        "--decoder_id",
        type=str,
        required=False,
        help="Optional pretrained decoder model ID (if different from encoder)",
    )
    parser.add_argument(
        "--block_size", type=int, default=1024, help="Block size for model input"
    )
    parser.add_argument(
        "--min_prefix_length",
        type=int,
        default=10,
        help="Minimum length of prefix in tokens",
    )
    parser.add_argument(
        "--max_prefix_ratio",
        type=float,
        default=0.7,
        help="Maximum ratio of prefix to total text length",
    )
    parser.add_argument(
        "--batch_size", type=int, default=8, help="Batch size for training"
    )
    parser.add_argument(
        "--num_epochs", type=int, default=10, help="Number of epochs to train for"
    )
    parser.add_argument(
        "--learning_rate", type=float, default=5e-5, help="Learning rate for training"
    )
    parser.add_argument(
        "--weight_decay", type=float, default=1e-4, help="Weight decay for training"
    )
    parser.add_argument(
        "--save_path",
        type=str,
        default="flow_matching_model",
        help="Path to save the trained model",
    )
    parser.add_argument(
        "--log_interval", type=int, default=100, help="Log interval for training"
    )
    parser.add_argument(
        "--eval_interval",
        type=int,
        default=1000,
        help="Evaluation interval for training",
    )
    parser.add_argument(
        "--checkpoint_interval",
        type=int,
        default=5000,
        help="Checkpoint interval for training",
    )
    parser.add_argument(
        "--use_wandb", action="store_true", help="Whether to use wandb for logging"
    )
    parser.add_argument(
        "--wandb_project",
        type=str,
        default="latent-flow-matching",
        help="Wandb project name",
    )
    parser.add_argument("--wandb_name", type=str, default=None, help="Wandb run name")
    parser.add_argument(
        "--num_samples",
        type=int,
        default=4,
        help="Number of samples to generate during evaluation",
    )
    parser.add_argument(
        "--seed", type=int, default=42, help="Random seed for reproducibility"
    )
    parser.add_argument(
        "--dtype",
        type=str,
        default="bfloat16",
        help="Data type for model parameters",
    )
    parser.add_argument(
        "--repeat_per_encode_pass",
        type=int,
        default=100,
        help="Number of times to sample different timesteps per encoded batch",
    )

    parser.add_argument(
        "--use_lora",
        action="store_true",
        help="Whether to use LoRA for training",
    )

    parser.add_argument(
        "--lora_r",
        type=int,
        default=128,
        help="LoRA rank",
    )

    parser.add_argument(
        "--lora_alpha",
        type=int,
        default=256,
        help="LoRA alpha",
    )

    parser.add_argument(
        "--lora_dropout",
        type=float,
        default=0.05,
        help="LoRA dropout",
    )

    parser.add_argument(
        "--lora_target_modules",
        type=str,
        default="q_proj,v_proj,k_proj,o_proj",
        help="LoRA target modules",
    )

    parser.add_argument(
        "--warmup_steps",
        type=int,
        default=100,
        help="Number of warmup steps",
    )

    parser.add_argument(
        "--max_steps",
        type=int,
        default=1000,
        help="Maximum number of steps",
    )

    return parser.parse_args()


def collate_fn(batch):
    """
    Custom collate function for batching examples.
    """
    prefixes = torch.stack([item["prefix"] for item in batch])
    suffixes = torch.stack([item["suffix"] for item in batch])

    return {
        "prefix": prefixes,
        "suffix": suffixes,
    }


def train_one_epoch(
    model,
    encoder,
    decoder,
    dataloader,
    optimizer,
    tokenizer,
    device,
    epoch,
    args,
):
    model.train()
    total_loss = 0
    total_steps = len(dataloader) * args.repeat_per_encode_pass

    # Create a single progress bar for all steps (batch × repeats)
    progress_bar = tqdm(
        total=total_steps,
        desc=f"Epoch {epoch}",
    )
    timesteps_histogram = []
    for batch_idx, batch in enumerate(dataloader):
        # Move batch to device
        prefix_tokens = batch["prefix"].to(device)
        suffix_tokens = batch["suffix"].to(device)
        print(suffix_tokens[0])
        # Encode suffix tokens to latent space (target latents)
        with torch.no_grad():
            suffix_latents, _, _ = encoder(suffix_tokens, tokenizer.pad_token_id)
            suffix_latents = (suffix_latents - VAE_SHIFT) / VAE_SCALE
            vae_mean = suffix_latents.mean().item()
            vae_std = suffix_latents.std().item()

        # Sample random timesteps
        batch_size = prefix_tokens.size(0)
        optimizer.zero_grad()  # Zero gradients once before the loop
        accumulated_loss = 0

        if batch_idx < args.warmup_steps:
            repeat_per_encode_pass = 1
        else:
            repeat_per_encode_pass = args.repeat_per_encode_pass

        for i in range(repeat_per_encode_pass):
            timesteps = torch.randint(
                1, args.max_steps + 1, (batch_size,), device=device
            ).tolist()

            timesteps_histogram.extend(timesteps)

            # Log wandb timesteps
            wandb.log(
                {
                    "train/timesteps": wandb.Histogram(timesteps_histogram),
                }
            )

            # Calculate flow matching loss
            loss = model.get_loss(prefix_tokens, suffix_latents, timesteps)
            # Scale loss by the number of accumulation steps to maintain effective learning rate
            scaled_loss = loss / repeat_per_encode_pass
            scaled_loss.backward()  # Gradients will accumulate across iterations

            accumulated_loss += loss.item()

            # Update progress bar for each repeat step
            progress_bar.update(1)
            progress_bar.set_postfix(
                {
                    "loss": loss.item(),
                    "vae_mean": vae_mean,
                    "vae_std": vae_std,
                }
            )

        # Average loss for the batch
        avg_batch_loss = accumulated_loss / args.repeat_per_encode_pass
        optimizer.step()  # Step optimizer once after the loop

        # Logging - use batch_idx instead of step
        total_loss += avg_batch_loss

        # Log to wandb
        if args.use_wandb and batch_idx % args.log_interval == 0:
            global_step = batch_idx + epoch * len(dataloader)
            wandb.log(
                {
                    "train/loss": avg_batch_loss,
                    "train/epoch": epoch,
                    "train/step": global_step,
                }
            )

        # # Save checkpoint
        # if step % args.checkpoint_interval == 0 and step > 0:
        #     save_checkpoint(model, optimizer, epoch, step, args)

        # Run evaluation
        if batch_idx % args.eval_interval == 0:
            evaluate(
                model,
                encoder,
                decoder,
                dataloader,
                tokenizer,
                device,
                epoch,
                batch_idx,
                args,
            )

    avg_loss = total_loss / len(dataloader)
    return avg_loss


def evaluate(
    model,
    encoder,
    decoder,
    dataloader,
    tokenizer,
    device,
    epoch,
    step,
    args,
):
    """Evaluate the model by generating completions for sample prefixes"""
    model.eval()

    # Get a few examples
    eval_batch = next(iter(dataloader))
    prefix_tokens = eval_batch["prefix"][: args.num_samples].to(device)
    suffix_tokens = eval_batch["suffix"][: args.num_samples].to(device)

    # Decode true suffixes for reference
    true_suffixes = []
    for i in range(min(args.num_samples, len(suffix_tokens))):
        # Filter out padding tokens
        suffix = suffix_tokens[i]
        suffix = suffix[suffix != tokenizer.pad_token_id]
        true_suffix = tokenizer.decode(suffix, skip_special_tokens=True)
        true_suffixes.append(true_suffix)

    # Generate completions
    generated_suffixes = []
    latent_means = []
    latent_stds = []
    for i in range(min(args.num_samples, len(prefix_tokens))):
        prefix = prefix_tokens[i : i + 1]  # Keep batch dimension

        # Generate initial noise
        B, T, D = 1, model.n_gist_tokens, model.base_config.hidden_size
        initial_noise = torch.randn(B, T, D, device=device)

        # Sample using flow matching
        with torch.no_grad():
            predicted_latents = model.sample(
                input_ids=prefix, initial_noise=initial_noise, num_steps=100
            )
            predicted_latents = (predicted_latents * VAE_SCALE) + VAE_SHIFT
            # Track latent statistics
            latent_means.append(predicted_latents.mean().item())
            latent_stds.append(predicted_latents.std().item())
            # Decode using the decoder
            output_ids = decoder.generate(
                predicted_latents, max_new_tokens=50, temperature=0.0
            )

        # Decode generated text
        generated_suffix = tokenizer.decode(output_ids[0], skip_special_tokens=True)
        generated_suffixes.append(generated_suffix)

    # Decode prefixes for context
    prefixes = []
    for i in range(min(args.num_samples, len(prefix_tokens))):
        # Filter out padding tokens
        prefix = prefix_tokens[i]
        prefix = prefix[prefix != tokenizer.pad_token_id]
        prefix_text = tokenizer.decode(prefix, skip_special_tokens=True)
        prefixes.append(prefix_text)

    # Log examples
    if args.use_wandb:
        examples = []
        for i in range(len(prefixes)):
            examples.append(
                {
                    "prefix": prefixes[i],
                    "true_suffix": true_suffixes[i],
                    "generated_suffix": generated_suffixes[i],
                }
            )

        # Convert to pandas DataFrame
        examples_df = pd.DataFrame(examples)

        wandb.log(
            {
                "examples": wandb.Table(
                    dataframe=examples_df,
                    columns=["prefix", "true_suffix", "generated_suffix"],
                ),
                "eval/epoch": epoch,
                "eval/step": step,
            }
        )

    # Log latent statistics
    if args.use_wandb:
        wandb.log(
            {
                "eval/latent_mean": np.mean(latent_means),
                "eval/latent_std": np.mean(latent_stds),
                "eval/latent_distribution": wandb.Histogram(
                    predicted_latents.float().cpu().numpy()
                ),
            }
        )

    # Print a few examples
    print("\n=== EVALUATION EXAMPLES ===")
    for i in range(len(prefixes)):
        print(f"Prefix: {prefixes[i]}")
        print(f"True suffix: {true_suffixes[i]}")
        print(f"Generated: {generated_suffixes[i]}")
        print("---")


def save_checkpoint(model, optimizer, epoch, step, args):
    """Save model checkpoint"""
    checkpoint_dir = f"{args.save_path}/checkpoints"
    os.makedirs(checkpoint_dir, exist_ok=True)

    checkpoint_path = f"{checkpoint_dir}/checkpoint_epoch{epoch}_step{step}.pt"
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "epoch": epoch,
            "step": step,
        },
        checkpoint_path,
    )

    print(f"Saved checkpoint to {checkpoint_path}")


def push_to_hub(model, args):
    """Push trained model to HuggingFace Hub"""
    from huggingface_hub import HfApi

    # Save model locally first
    os.makedirs(f"{args.save_path}/final", exist_ok=True)
    model_path = f"{args.save_path}/final/model.pt"
    torch.save(model.state_dict(), model_path)

    # Push to Hub
    api = HfApi()
    repo_id = (
        f"{args.wandb_project}/{args.wandb_name}"
        if args.wandb_name
        else f"{args.wandb_project}/flow-matching-model"
    )

    # Create repo if it doesn't exist
    try:
        api.create_repo(repo_id, exist_ok=True)
    except:
        print(f"Repository {repo_id} already exists")

    # Upload model
    api.upload_file(
        path_or_fileobj=model_path,
        path_in_repo="model.pt",
        repo_id=repo_id,
    )

    # Upload config
    import json

    config = {
        "model_name": args.model_name,
        "encoder_id": args.encoder_id,
        "decoder_id": args.decoder_id or args.encoder_id,
        "task": "text_completion",
    }

    config_path = f"{args.save_path}/final/config.json"
    with open(config_path, "w") as f:
        json.dump(config, f)

    api.upload_file(
        path_or_fileobj=config_path,
        path_in_repo="config.json",
        repo_id=repo_id,
    )

    print(f"Pushed model to {repo_id}")


def main():
    args = parse_args()

    # Set random seeds for reproducibility
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    torch_dtype = torch.bfloat16 if args.dtype == "bfloat16" else torch.float32
    # Initialize wandb
    if args.use_wandb:
        wandb.init(
            project=args.wandb_project,
            name=args.wandb_name,
            config=vars(args),
        )

    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load encoder model
    print(f"Loading encoder model {args.encoder_id}...")
    encoder = LatentEncoder.from_pretrained(
        args.encoder_id, device=device, torch_dtype=torch_dtype
    )
    encoder.eval()  # We don't train the encoder, only use it to generate latents

    # Load decoder model (for evaluation)
    decoder_id = args.decoder_id if args.decoder_id else args.encoder_id
    print(f"Loading decoder model {decoder_id}...")

    encoder_config = encoder.latent_config

    # Create decoder
    decoder = LatentDecoder(
        model_name=decoder_id,
        n_gist_tokens=encoder_config["n_gist_tokens"],
        block_size=encoder_config["block_size"],
        torch_dtype=torch_dtype,
    )
    decoder.to(device)
    decoder.eval()

    # Create flow matching model
    print(f"Creating flow matching model based on {args.model_name}...")
    flow_model = GPTLatentFlowMatching(
        model_name=args.model_name,
        n_gist_tokens=encoder_config["n_gist_tokens"],
        block_size=encoder_config["block_size"],
        max_steps=args.max_steps,
        device=device,
        torch_dtype=torch_dtype,
        use_lora=args.use_lora,
        lora_r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        lora_target_modules=args.lora_target_modules.split(","),
    )
    flow_model.to(device)

    # Get tokenizer from encoder
    tokenizer = encoder.tokenizer

    # Create dataset and dataloader
    print("Creating dataset...")
    dataset = TextCompletionDataset(
        dataset_id=args.dataset,
        split="train",
        tokenizer=tokenizer,
        block_size=encoder_config["block_size"],
        min_prefix_length=args.min_prefix_length,
        max_prefix_ratio=args.max_prefix_ratio,
    )

    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=4,
        collate_fn=collate_fn,
    )

    # Create optimizer
    optimizer = torch.optim.AdamW(
        [p for p in flow_model.parameters() if p.requires_grad],
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
    )

    # Create save directory
    os.makedirs(args.save_path, exist_ok=True)

    # Training loop
    for epoch in range(args.num_epochs):
        print(f"Starting epoch {epoch + 1}/{args.num_epochs}")
        avg_loss = train_one_epoch(
            flow_model,
            encoder,
            decoder,
            dataloader,
            optimizer,
            tokenizer,
            device,
            epoch,
            args,
        )

        print(f"Epoch {epoch + 1} completed. Average loss: {avg_loss:.4f}")

        # Save model after each epoch
        # save_checkpoint(flow_model, optimizer, epoch, 0, args)

    # Save final model
    if args.use_wandb:
        push_to_hub(flow_model, args)

    print("Training completed!")


if __name__ == "__main__":
    main()
