from dataclasses import dataclass
import torch
from latent_llm.data.text_dataset import TextDataset, RandomTextDataset
from latent_llm.models.gpt_latent import LatentEncoder, LatentDecoder
from torch.utils.data import DataLoader
from transformers import AutoTokenizer
import wandb
import logging
from rich.logging import RichHandler
from accelerate import Accelerator
import time
import argparse

torch.set_float32_matmul_precision("high")

accelerator = Accelerator(mixed_precision="bf16")


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
    parser.add_argument("--model_name", type=str, default="HuggingFaceTB/SmolLM2-135M")
    parser.add_argument("--dataset_type", type=str, default="text")
    parser.add_argument(
        "--dataset_id", type=str, default="BEE-spoke-data/fineweb-100k_en-med"
    )
    parser.add_argument("--split", type=str, default="train")
    parser.add_argument("--block_size", type=int, default=256)
    parser.add_argument("--n_gist_tokens", type=int, default=256)
    parser.add_argument(
        "--hub_repo_id", type=str, default="toilaluan/smol-lm-2-135m-latent-encoder"
    )
    parser.add_argument("--learning_rate", type=float, default=2.5e-4)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--max_grad_norm", type=float, default=1.0)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--num_workers", type=int, default=8)
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
    parser.add_argument("--log_interval", type=int, default=10)
    parser.add_argument("--validating_interval", type=int, default=100)
    parser.add_argument("--save_interval", type=int, default=1_000)
    parser.add_argument("--max_new_tokens", type=int, default=512)
    parser.add_argument("--training_steps", type=int, default=100_000)
    parser.add_argument("--wandb_project", type=str, default="latent-llm")
    parser.add_argument("--limit", type=int, default=-1)
    return parser.parse_args()


def main():
    args = parse_args()

    wandb.init(project=args.wandb_project)

    print("--- Training Config ---")
    logger.info(args)
    print("---")

    ENCODER = LatentEncoder(
        model_name=args.model_name,
        n_gist_tokens=args.n_gist_tokens,
        block_size=args.block_size,
    )
    DECODER = LatentDecoder(
        model_name=args.model_name,
        n_gist_tokens=args.n_gist_tokens,
        block_size=args.block_size,
    )

    TOKENIZER = AutoTokenizer.from_pretrained(args.model_name)
    # Add new pad token
    TOKENIZER.add_special_tokens({"pad_token": "<|pad|>"})
    TOKENIZER.push_to_hub(args.hub_repo_id + "-encoder")
    logger.info(f"pad_token: {TOKENIZER.pad_token}: {TOKENIZER.pad_token_id}")
    logger.info(f"eos_token: {TOKENIZER.eos_token}: {TOKENIZER.eos_token_id}")
    if args.dataset_type == "text":
        DATASET = TextDataset(
            dataset_id=args.dataset_id,
            split=args.split,
            block_size=args.block_size,
            model_name=args.model_name,
        )
    else:
        DATASET = RandomTextDataset(
            model_name=args.model_name,
            block_size=args.block_size,
        )
    logger.info(f"sample: {DATASET[0]}")
    DATALOADER = DataLoader(
        DATASET,
        batch_size=args.batch_size,
        shuffle=True,
    )
    DATALOADER = cycle(DATALOADER)

    def training_step(batch: torch.Tensor) -> torch.Tensor:
        input_ids = batch.to(accelerator.device)
        labels = batch.to(accelerator.device)
        mem_embeds, kl_loss, _ = ENCODER(input_ids, pad_token_id=TOKENIZER.pad_token_id)
        logits, loss, token_accuracy = DECODER(
            input_ids, mem_embeds, labels=labels, ignore_index=TOKENIZER.pad_token_id
        )
        # Combine reconstruction loss with KL divergence loss
        total_loss = loss + kl_loss
        return total_loss, loss, kl_loss, mem_embeds, input_ids, token_accuracy

    def count_parameters(model):
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        return trainable_params, total_params

    current_step = 0

    ENCODER.train()
    DECODER.train()

    # ENCODER.forward = torch.compile(ENCODER.forward)
    # DECODER.forward = torch.compile(DECODER.forward)

    # Log the number of parameters
    encoder_trainable_params, encoder_total_params = count_parameters(ENCODER)
    decoder_trainable_params, decoder_total_params = count_parameters(DECODER)

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

    TRAIN_PARAMS = []
    TRAIN_PARAMS.extend([p for p in ENCODER.parameters() if p.requires_grad])
    TRAIN_PARAMS.extend([p for p in DECODER.parameters() if p.requires_grad])

    OPTIMIZER = torch.optim.AdamW(
        TRAIN_PARAMS,
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
    )

    ENCODER, DECODER, DATALOADER, OPTIMIZER = accelerator.prepare(
        ENCODER, DECODER, DATALOADER, OPTIMIZER
    )
    ENCODER.to(accelerator.device)
    DECODER.to(accelerator.device)

    PROCESSED_TOKENS = 0
    START_TIME = time.time()

    for batch in DATALOADER:
        OPTIMIZER.zero_grad()
        total_loss, rec_loss, kl_loss, mem_embeds, input_ids, token_accuracy = (
            training_step(batch)
        )
        mem_embeds_mean = mem_embeds.detach().mean()
        mem_embeds_std = mem_embeds.detach().std()
        wandb.log(
            {
                "train/total_loss": total_loss.detach().item(),
                "train/reconstruction_loss": rec_loss.detach().item(),
                "train/kl_loss": kl_loss.detach().item(),
                "train/token_accuracy": token_accuracy.detach().item(),
                "train/mem_embeds_mean": mem_embeds_mean.detach().item(),
                "train/mem_embeds_std": mem_embeds_std.detach().item(),
            }
        )
        accelerator.backward(total_loss)
        torch.nn.utils.clip_grad_norm_(TRAIN_PARAMS, 1.0)
        OPTIMIZER.step()
        PROCESSED_TOKENS += args.block_size * args.batch_size
        TOKEN_PER_SECOND = PROCESSED_TOKENS / (time.time() - START_TIME)

        if current_step % args.log_interval == 0:
            logger.info(
                f"[{current_step}/{args.training_steps}] total_loss: {total_loss.detach().item():.4f}; rec_loss: {rec_loss.detach().item():.4f}; kl_loss: {kl_loss.detach().item():.4f}; token_accuracy: {token_accuracy.detach().item():.4f}; {TOKEN_PER_SECOND:.2f} tokens/s (processed {PROCESSED_TOKENS} tokens)"
            )
            logger.info(
                f"mem_embeds_mean: {mem_embeds_mean.detach().item():.4f}; mem_embeds_std: {mem_embeds_std.detach().item():.4f}"
            )

        if args.save_interval > 0 and current_step % args.save_interval == 0:
            logger.info("Saving to hub...")
            try:
                # Unwrap models before pushing to hub
                unwrapped_encoder = accelerator.unwrap_model(ENCODER)
                unwrapped_encoder.push_to_hub(args.hub_repo_id + "-encoder")
                unwrapped_decoder = accelerator.unwrap_model(DECODER)
                unwrapped_decoder.push_to_hub(args.hub_repo_id + "-decoder")
            except Exception as e:
                logger.error(f"Error pushing to hub: {e}")

        if current_step % args.validating_interval == 0:
            logger.info("Generating...")
            ENCODER.eval()
            DECODER.eval()
            mem_mean = mem_embeds.detach().mean()
            mem_std = mem_embeds.detach().std()
            logger.info(
                f"mem_mean: {mem_mean.detach().item():.4f}; mem_std: {mem_std.detach().item():.4f}"
            )
            with torch.no_grad():
                batch = next(iter(DATALOADER))
                generated_ids = DECODER.generate(
                    mem_embeds[:1, :, :],
                    max_new_tokens=args.max_new_tokens,
                )
                completion = TOKENIZER.decode(generated_ids[0])
                label = TOKENIZER.decode(input_ids[0, :])
                # Log completion and input_ids
                wandb.log(
                    {
                        "train/completion": wandb.Table(
                            columns=["Type", "Text"],
                            data=[["Completion", completion], ["Label", label]],
                        ),
                    }
                )
                logger.info(
                    f"[{current_step}/{args.training_steps}] completion: {completion[:32]}..."
                )
                logger.info(
                    f"[{current_step}/{args.training_steps}] label: {label[:32]}..."
                )
            ENCODER.train()
            DECODER.train()
            START_TIME = time.time()
            current_step += 1

        if current_step >= args.training_steps:
            break

        current_step += 1


if __name__ == "__main__":
    main()
