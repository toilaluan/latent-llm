import argparse
import os
import torch
from latent_llm.models.latent_diffusion import train_latent_diffusion
import logging
from rich.logging import RichHandler

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(message)s", datefmt="[%X]", handlers=[RichHandler()]
)
logger = logging.getLogger(__name__)


def parse_args():
    parser = argparse.ArgumentParser(description="Train a latent diffusion model")

    # Required arguments
    parser.add_argument(
        "--encoder_model_id",
        type=str,
        required=True,
        help="HuggingFace model ID for the latent encoder",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        required=True,
        help="Directory to save model checkpoints",
    )
    parser.add_argument(
        "--dataset_path",
        type=str,
        required=True,
        help="Path to Hugging Face dataset containing text samples",
    )

    # Optional arguments
    parser.add_argument(
        "--num_epochs",
        type=int,
        default=100,
        help="Number of training epochs",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=16,
        help="Batch size for training",
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=1e-4,
        help="Learning rate",
    )
    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=1,
        help="Number of steps for gradient accumulation",
    )
    parser.add_argument(
        "--lr_warmup_steps",
        type=int,
        default=500,
        help="Number of warmup steps for learning rate scheduler",
    )
    parser.add_argument(
        "--save_image_epochs",
        type=int,
        default=10,
        help="Save sample generated latents every N epochs",
    )
    parser.add_argument(
        "--save_model_epochs",
        type=int,
        default=50,
        help="Save model checkpoint every N epochs",
    )
    parser.add_argument(
        "--mixed_precision",
        type=str,
        default="fp16",
        choices=["no", "fp16", "bf16"],
        help="Mixed precision type",
    )
    parser.add_argument(
        "--use_hf_unet",
        action="store_true",
        help="Use HuggingFace UNet instead of custom MLP",
    )
    parser.add_argument(
        "--push_to_hub",
        action="store_true",
        help="Push model to Hugging Face Hub",
    )
    parser.add_argument(
        "--hub_model_id",
        type=str,
        default=None,
        help="Model ID for pushing to Hub",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed",
    )

    return parser.parse_args()


def main():
    args = parse_args()

    # Set random seed
    torch.manual_seed(args.seed)

    logger.info(f"Training latent diffusion model with the following parameters:")
    for arg, value in vars(args).items():
        logger.info(f"  {arg}: {value}")

    # Train the model
    model = train_latent_diffusion(
        encoder_model_id=args.encoder_model_id,
        output_dir=args.output_dir,
        dataset_path=args.dataset_path,
        num_epochs=args.num_epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        lr_warmup_steps=args.lr_warmup_steps,
        save_image_epochs=args.save_image_epochs,
        save_model_epochs=args.save_model_epochs,
        mixed_precision=args.mixed_precision,
        use_hf_unet=args.use_hf_unet,
        push_to_hub=args.push_to_hub,
        hub_model_id=args.hub_model_id,
        seed=args.seed,
    )

    logger.info(f"Training completed. Model saved to {args.output_dir}")


if __name__ == "__main__":
    main()
