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


@dataclass
class TrainingConfig:
    model_name: str = "HuggingFaceTB/SmolLM2-135M"
    dataset_id: str = "BEE-spoke-data/fineweb-100k_en-med"
    split: str = "train"
    dataset_type: str = "text"
    block_size: int = 256
    n_gist_tokens: int = 256
    n_ae_tokens: int = 1
    hub_repo_id: str = "toilaluan/smol-lm-2-135m-latent-encoder"
    learning_rate: float = 1e-4
    weight_decay: float = 1e-4
    max_grad_norm: float = 1.0
    max_epochs: int = 10
    batch_size: int = 1
    num_workers: int = 8
    seed: int = 42
    device: str = (
        "cuda"
        if torch.cuda.is_available()
        else "mps" if torch.backends.mps.is_available() else "cpu"
    )
    precision: str = "16-mixed" if device == "cuda" else "32"
    log_interval: int = 10
    validating_interval: int = 100
    save_interval: int = 1_000
    max_new_tokens: int = 512
    training_steps: int = 100_000
    wandb_project: str = "latent-llm"
    limit: int = -1
    freeze_decoder: bool = True
    # LoRA parameters
    use_lora: bool = False
    lora_r: int = 8
    lora_alpha: int = 16
    lora_dropout: float = 0.05
    lora_target_modules: str = "q_proj,v_proj"  # Comma-separated list of module names


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
    parser.add_argument("--n_ae_tokens", type=int, default=1)
    parser.add_argument(
        "--hub_repo_id", type=str, default="toilaluan/smol-lm-2-135m-latent"
    )
    parser.add_argument("--learning_rate", type=float, default=1e-4)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--max_grad_norm", type=float, default=1.0)
    parser.add_argument("--max_epochs", type=int, default=10)
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
    parser.add_argument(
        "--precision",
        type=str,
        default="16-mixed" if torch.cuda.is_available() else "32",
    )
    parser.add_argument("--log_interval", type=int, default=10)
    parser.add_argument("--validating_interval", type=int, default=100)
    parser.add_argument("--save_interval", type=int, default=1_000)
    parser.add_argument("--max_new_tokens", type=int, default=512)
    parser.add_argument("--training_steps", type=int, default=100_000)
    parser.add_argument("--wandb_project", type=str, default="latent-llm")
    parser.add_argument("--limit", type=int, default=-1)
    parser.add_argument("--freeze_decoder", default=False, action="store_true")
    # LoRA arguments
    parser.add_argument(
        "--use_lora",
        default=False,
        action="store_true",
        help="Use PEFT LoRA for fine-tuning",
    )
    parser.add_argument(
        "--lora_r", type=int, default=8, help="Rank of LoRA adaptation matrices"
    )
    parser.add_argument(
        "--lora_alpha", type=int, default=16, help="LoRA scaling factor"
    )
    parser.add_argument(
        "--lora_dropout",
        type=float,
        default=0.05,
        help="Dropout probability for LoRA layers",
    )
    parser.add_argument(
        "--lora_target_modules",
        type=str,
        default="q_proj,v_proj",
        help="Comma-separated list of modules to apply LoRA to",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    config = TrainingConfig(
        model_name=args.model_name,
        dataset_id=args.dataset_id,
        split=args.split,
        dataset_type=args.dataset_type,
        block_size=args.block_size,
        n_gist_tokens=args.n_gist_tokens,
        n_ae_tokens=args.n_ae_tokens,
        hub_repo_id=args.hub_repo_id,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        max_grad_norm=args.max_grad_norm,
        max_epochs=args.max_epochs,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        seed=args.seed,
        device=args.device,
        precision=args.precision,
        log_interval=args.log_interval,
        validating_interval=args.validating_interval,
        save_interval=args.save_interval,
        max_new_tokens=args.max_new_tokens,
        training_steps=args.training_steps,
        wandb_project=args.wandb_project,
        limit=args.limit,
        freeze_decoder=args.freeze_decoder,
        use_lora=args.use_lora,
        lora_r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        lora_target_modules=args.lora_target_modules,
    )

    wandb.init(project=config.wandb_project)

    print("--- Training Config ---")
    logger.info(config)
    print("---")

    ENCODER = LatentEncoder(
        config.model_name,
        config.n_gist_tokens,
        config.n_ae_tokens,
        config.block_size,
        use_lora=config.use_lora,
        lora_r=config.lora_r,
        lora_alpha=config.lora_alpha,
        lora_dropout=config.lora_dropout,
        lora_target_modules=(
            config.lora_target_modules.split(",")
            if config.lora_target_modules
            else None
        ),
    )
    DECODER = LatentDecoder(
        config.model_name, config.n_gist_tokens, config.n_ae_tokens, config.block_size
    )

    TOKENIZER = AutoTokenizer.from_pretrained(config.model_name)
    TOKENIZER.pad_token = TOKENIZER.eos_token

    if config.dataset_type == "text":
        DATASET = TextDataset(
            dataset_id=config.dataset_id,
            split=config.split,
            block_size=config.block_size,
            model_name=config.model_name,
            limit=config.limit,
        )
    else:
        DATASET = RandomTextDataset(
            model_name=config.model_name,
            block_size=config.block_size,
        )

    DATALOADER = DataLoader(
        DATASET,
        batch_size=config.batch_size,
        shuffle=True,
    )
    DATALOADER = cycle(DATALOADER)

    def training_step(batch: torch.Tensor) -> torch.Tensor:
        input_ids = batch.to(accelerator.device)
        labels = batch.to(accelerator.device)
        mem_embeds = ENCODER(input_ids, pad_token_id=TOKENIZER.pad_token_id)
        logits, loss, token_accuracy = DECODER(
            input_ids, mem_embeds, labels=labels, ignore_index=TOKENIZER.pad_token_id
        )
        return loss, mem_embeds, input_ids, token_accuracy

    def count_parameters(model):
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        return trainable_params, total_params

    current_step = 0

    ENCODER.train()
    DECODER.train()

    if config.freeze_decoder:
        for param in DECODER.parameters():
            param.requires_grad = False

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

    OPTIMIZER = torch.optim.AdamW(
        ENCODER.get_trainable_parameters(),  # Use the method to get only trainable parameters
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
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
        loss, mem_embeds, input_ids, token_accuracy = training_step(batch)
        wandb.log({"train/loss": loss.item()})
        wandb.log({"train/token_accuracy": token_accuracy.item()})
        accelerator.backward(loss)
        OPTIMIZER.step()
        PROCESSED_TOKENS += config.block_size * config.batch_size
        TOKEN_PER_SECOND = PROCESSED_TOKENS / (time.time() - START_TIME)

        if current_step % config.log_interval == 0:
            logger.info(
                f"[{current_step}/{config.training_steps}] loss: {loss.item():.4f}; token_accuracy: {token_accuracy.item():.4f}; {TOKEN_PER_SECOND:.2f} tokens/s (processed {PROCESSED_TOKENS} tokens)"
            )

        if config.save_interval > 0 and current_step % config.save_interval == 0:
            logger.info("Saving to hub...")
            try:
                # Unwrap models before pushing to hub
                unwrapped_encoder = accelerator.unwrap_model(ENCODER)
                unwrapped_decoder = accelerator.unwrap_model(DECODER)
                unwrapped_encoder.push_to_hub(config.hub_repo_id + "-encoder")
                unwrapped_decoder.push_to_hub(config.hub_repo_id + "-decoder")
            except Exception as e:
                logger.error(f"Error pushing to hub: {e}")

        if current_step % config.validating_interval == 0:
            logger.info("Generating...")
            ENCODER.eval()
            DECODER.eval()
            with torch.no_grad():
                batch = next(iter(DATALOADER))
                generated_ids = DECODER.generate(
                    mem_embeds[:1, :, :],
                    input_ids[:1, :1],  # Seed token
                    max_new_tokens=config.max_new_tokens,
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
                    f"[{current_step}/{config.training_steps}] completion: {completion[:32]}..."
                )
                logger.info(
                    f"[{current_step}/{config.training_steps}] label: {label[:32]}..."
                )
            ENCODER.train()
            DECODER.train()
            START_TIME = time.time()
            current_step += 1

        if current_step >= config.training_steps:
            break


if __name__ == "__main__":
    main()
