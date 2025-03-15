import torch
from latent_llm.data.text_dataset import (
    TextDataset,
    RandomTokenDataset,
)
from latent_llm.models.gpt_latent import LatentEncoder, LatentDecoder
from torch.utils.data import DataLoader
from transformers import AutoTokenizer
import wandb
import logging
from rich.logging import RichHandler
import time
import argparse

DEVICE = (
    "cuda"
    if torch.cuda.is_available()
    else "mps" if torch.backends.mps.is_available() else "cpu"
)


def cycle(dataloader):
    while True:
        for batch in dataloader:
            yield batch


# Update logging configuration to use RichHandler
logging.basicConfig(
    level=logging.INFO, format="%(message)s", datefmt="[%X]", handlers=[RichHandler()]
)

logger = logging.getLogger(__name__)


def parse_args():
    parser = argparse.ArgumentParser(description="Training script for latent LLM.")
    parser.add_argument("--model-name", type=str, default="HuggingFaceTB/SmolLM2-135M")
    parser.add_argument("--dataset-type", type=str, default="text")
    parser.add_argument(
        "--dataset-id", type=str, default="BEE-spoke-data/fineweb-100k_en-med"
    )
    parser.add_argument(
        "--val-samples", type=int, default=10, help="Number of validation samples"
    )
    parser.add_argument("--block-size", type=int, default=256)
    parser.add_argument("--latent-size", type=int, default=256)
    parser.add_argument(
        "--hub-repo-id", type=str, default="toilaluan/smol-lm-2-135m-latent-encoder"
    )
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--max-grad-norm", type=float, default=1.0)
    parser.add_argument("--use-grad-norm", action="store_true", default=False)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--device",
        type=str,
        default=(
            "cuda"
            if torch.cuda.is_available()
            else "mps" if torch.backends.mps.is_available() else "cpu"
        ),
    )
    parser.add_argument("--log-interval", type=int, default=10)
    parser.add_argument("--validating-interval", type=int, default=100)
    parser.add_argument("--save-interval", type=int, default=1_000)
    parser.add_argument("--training-steps", type=int, default=100_000)
    parser.add_argument("--wandb-project", type=str, default="latent-llm")
    parser.add_argument(
        "--kl-weight",
        type=float,
        default=0.00001,
        help="Weight for KL divergence loss",
    )
    parser.add_argument(
        "--attn-implementation",
        type=str,
        default="flash_attention_2",
        help="Attention implementation to use",
    )
    return parser.parse_args()


def setup_models(args):
    """Initialize and configure encoder and decoder models."""
    encoder = LatentEncoder(
        model_name=args.model_name,
        latent_size=args.latent_size,
        block_size=args.block_size,
        kl_weight=args.kl_weight,
        attn_implementation=args.attn_implementation,
    )
    decoder = LatentDecoder(
        model_name=args.model_name,
        latent_size=args.latent_size,
        block_size=args.block_size,
        attn_implementation=args.attn_implementation,
    )

    return encoder, decoder


def setup_tokenizer(args):
    """Setup and configure the tokenizer."""
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    # Add new pad token
    tokenizer.add_special_tokens({"pad_token": "<|pad|>"})
    tokenizer.push_to_hub(args.hub_repo_id + "-encoder")
    logger.info(f"pad_token: {tokenizer.pad_token}: {tokenizer.pad_token_id}")
    logger.info(f"eos_token: {tokenizer.eos_token}: {tokenizer.eos_token_id}")
    return tokenizer


def setup_datasets(args, tokenizer):
    """Setup training and validation datasets."""
    # Always use RandomTokenDataset for training
    train_dataset = RandomTokenDataset(
        model_name=args.model_name,
        block_size=args.block_size,
    )
    logger.info(f"Training sample: {train_dataset[0]}")
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
    )
    train_dataloader = cycle(train_dataloader)

    val_dataset = RandomTokenDataset(
        model_name=args.model_name,
        block_size=args.block_size,
    )
    val_dataloader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=True,
    )
    val_dataloader = cycle(val_dataloader)

    return train_dataloader, val_dataloader


def calculate_completion_accuracy(generated_ids, target_ids):
    """Calculate token-level accuracy between generated and target sequences"""
    min_len = min(len(generated_ids), len(target_ids))
    matches = sum(1 for i in range(min_len) if generated_ids[i] == target_ids[i])
    return matches / min_len if min_len > 0 else 0.0


def training_step(encoder, decoder, batch, tokenizer, device):
    """Perform a single training step."""
    input_ids = batch.to(device)
    labels = batch.to(device)
    latent_embeds, kl_loss, latents = encoder(
        input_ids, pad_token_id=tokenizer.pad_token_id
    )
    logits, loss, token_accuracy = decoder(
        input_ids, latent_embeds, labels=labels, ignore_index=tokenizer.pad_token_id
    )
    # Combine reconstruction loss with KL divergence loss
    total_loss = loss + kl_loss
    return total_loss, loss, kl_loss, latent_embeds, input_ids, token_accuracy


def count_parameters(model):
    """Count total and trainable parameters in a model."""
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return trainable_params, total_params


def log_parameter_counts(encoder, decoder):
    """Log parameter counts for encoder and decoder."""
    encoder_trainable_params, encoder_total_params = count_parameters(encoder)
    decoder_trainable_params, decoder_total_params = count_parameters(decoder)

    logger.info(
        f"Encoder: {encoder_trainable_params:,} trainable / {encoder_total_params:,} total parameters "
        f"({encoder_trainable_params/encoder_total_params:.2%})"
    )
    logger.info(
        f"Decoder: {decoder_trainable_params:,} trainable / {decoder_total_params:,} total parameters "
        f"({decoder_trainable_params/decoder_total_params:.2%})"
    )

    wandb.log(
        {
            "encoder/trainable_params": encoder_trainable_params,
            "encoder/total_params": encoder_total_params,
            "decoder/trainable_params": decoder_trainable_params,
            "decoder/total_params": decoder_total_params,
        }
    )

    return (
        encoder_trainable_params,
        encoder_total_params,
        decoder_trainable_params,
        decoder_total_params,
    )


def validate(encoder, decoder, val_dataloader, tokenizer, args):
    """Run validation and log metrics."""
    encoder.eval()
    decoder.eval()
    n_samples = 4
    val_total_loss = 0.0
    val_rec_loss = 0.0
    val_kl_loss = 0.0
    val_token_accuracy = 0.0
    val_completion_accuracy = 0.0
    val_completion_accuracy_rep = 0.0
    i = 0
    with torch.no_grad():
        # Process validation samples
        for batch in val_dataloader:
            if i >= n_samples:
                break
            i += 1
            val_sample = batch.to(DEVICE)

            # Get latent representation
            rep_latent_embeds, kl_loss, latent_embeds = encoder(
                val_sample, pad_token_id=tokenizer.pad_token_id
            )
            # Calculate reconstruction loss
            logits, loss, token_acc = decoder(
                val_sample,
                rep_latent_embeds,
                labels=val_sample,
                ignore_index=tokenizer.pad_token_id,
            )
            latent_mean = latent_embeds[0, :, :].mean()
            latent_std = latent_embeds[0, :, :].std()
            logger.info(
                f"latent_mean: {latent_mean.item():.4f}; latent_std: {latent_std.item():.4f}"
            )

            # Generate completion from reparametrized latent embeddings
            rep_generated_ids = decoder.generate(
                rep_latent_embeds[:1],
                max_new_tokens=encoder.block_size,
            )[0].tolist()

            # Calculate completion accuracy
            target_ids = val_sample[0].tolist()
            sample_completion_acc_rep = calculate_completion_accuracy(
                rep_generated_ids, target_ids
            )

            # Generate completion from original latent embeddings
            generated_ids = decoder.generate(
                latent_embeds[:1],
                max_new_tokens=encoder.block_size,
            )[0].tolist()

            sample_completion_acc = calculate_completion_accuracy(
                generated_ids, target_ids
            )

            # Accumulate metrics
            val_total_loss += loss.item() + kl_loss.item()
            val_rec_loss += loss.item()
            val_kl_loss += kl_loss.item()
            val_token_accuracy += token_acc.item()
            val_completion_accuracy += sample_completion_acc
            val_completion_accuracy_rep += sample_completion_acc_rep
        # Average metrics
        val_total_loss /= n_samples
        val_rec_loss /= n_samples
        val_kl_loss /= n_samples
        val_token_accuracy /= n_samples
        val_completion_accuracy /= n_samples
        val_completion_accuracy_rep /= n_samples
        # Log validation metrics
        wandb.log(
            {
                "val/total_loss": val_total_loss,
                "val/reconstruction_loss": val_rec_loss,
                "val/kl_loss": val_kl_loss,
                "val/token_accuracy": val_token_accuracy,
                "val/completion_accuracy": val_completion_accuracy,
                "val/completion_accuracy_rep": val_completion_accuracy_rep,
            }
        )

        # Log a sample completion
        completion = tokenizer.decode(generated_ids)
        completion_rep = tokenizer.decode(rep_generated_ids)
        label = tokenizer.decode(target_ids)
        wandb.log(
            {
                "val/completion": wandb.Table(
                    columns=["Type", "Text"],
                    data=[
                        ["Completion", completion],
                        ["Label", label],
                        ["Completion Rep", completion_rep],
                    ],
                ),
            }
        )

        logger.info(f"Validation metrics:")
        logger.info(f"  total_loss: {val_total_loss:.4f}")
        logger.info(f"  rec_loss: {val_rec_loss:.4f}")
        logger.info(f"  kl_loss: {val_kl_loss:.4f}")
        logger.info(f"  token_accuracy: {val_token_accuracy:.4f}")
        logger.info(f"  completion_accuracy: {val_completion_accuracy:.4f}")
        logger.info(f"  completion_accuracy_rep: {val_completion_accuracy_rep:.4f}")
        logger.info(f"  sample completion: {completion}...")
        logger.info(f"  sample label: {label}...")
        logger.info(f"  sample completion_rep: {completion_rep}...")
    encoder.train()
    decoder.train()

    return val_total_loss, val_completion_accuracy


def save_models(encoder, decoder, args):
    """Save models to Hugging Face Hub."""
    logger.info("Saving to hub...")
    try:
        # Unwrap models before pushing to hub
        encoder.push_to_hub(args.hub_repo_id + "-encoder")
        decoder.push_to_hub(args.hub_repo_id + "-decoder")
    except Exception as e:
        logger.error(f"Error pushing to hub: {e}")


def main():
    args = parse_args()

    wandb.init(project=args.wandb_project)

    print("--- Training Config ---")
    logger.info(args)
    print("---")

    # Setup models, tokenizer and datasets
    encoder, decoder = setup_models(args)
    tokenizer = setup_tokenizer(args)
    train_dataloader, val_dataloader = setup_datasets(args, tokenizer)

    # Log parameter counts
    log_parameter_counts(encoder, decoder)

    # Setup optimizer
    train_params = []
    train_params.extend([p for p in encoder.parameters() if p.requires_grad])
    train_params.extend([p for p in decoder.parameters() if p.requires_grad])

    optimizer = torch.optim.AdamW(
        train_params,
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
    )

    # Prepare for distributed training
    encoder.to(DEVICE)
    decoder.to(DEVICE)

    # Training loop
    current_step = 0
    processed_tokens = 0
    start_time = time.time()

    for batch in train_dataloader:
        optimizer.zero_grad()
        total_loss, rec_loss, kl_loss, latent_embeds, input_ids, token_accuracy = (
            training_step(encoder, decoder, batch, tokenizer, DEVICE)
        )
        latent_embeds_mean = latent_embeds[0, :, :].mean()
        latent_embeds_std = latent_embeds[0, :, :].std()
        wandb.log(
            {
                "train/total_loss": total_loss.item(),
                "train/reconstruction_loss": rec_loss.item(),
                "train/kl_loss": kl_loss.item(),
                "train/token_accuracy": token_accuracy.item(),
                "train/latent_embeds_mean": latent_embeds_mean.item(),
                "train/latent_embeds_std": latent_embeds_std.item(),
            }
        )
        total_loss.backward()
        if args.use_grad_norm:
            torch.nn.utils.clip_grad_norm_(train_params, args.max_grad_norm)
        optimizer.step()
        processed_tokens += args.block_size * args.batch_size
        token_per_second = processed_tokens / (time.time() - start_time)

        if current_step % args.log_interval == 0:
            logger.info(
                f"[{current_step}/{args.training_steps}] total_loss: {total_loss.item():.4f}; rec_loss: {rec_loss.item():.4f}; kl_loss: {kl_loss.item():.4f}; token_accuracy: {token_accuracy.item():.4f}; {token_per_second:.2f} tokens/s (processed {processed_tokens} tokens)"
            )
            logger.info(
                f"latent_embeds_mean: {latent_embeds_mean.item():.4f}; latent_embeds_std: {latent_embeds_std.item():.4f}"
            )

        if args.save_interval > 0 and current_step % args.save_interval == 0:
            save_models(encoder, decoder, args)

        if (
            args.validating_interval > 0
            and current_step % args.validating_interval == 0
        ):
            logger.info("Validating...")
            validate(encoder, decoder, val_dataloader, tokenizer, args)
            start_time = time.time()  # Reset timer after validation

        current_step += 1

        if current_step >= args.training_steps:
            break


if __name__ == "__main__":
    main()
