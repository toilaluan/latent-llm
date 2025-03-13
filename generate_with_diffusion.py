import argparse
import os
import torch
from latent_llm.models.latent_diffusion import LatentDiffusionPipeline
import logging
from rich.logging import RichHandler

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(message)s", datefmt="[%X]", handlers=[RichHandler()]
)
logger = logging.getLogger(__name__)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Generate text using latent diffusion model"
    )

    # Required arguments
    parser.add_argument(
        "--diffusion_model_path",
        type=str,
        required=True,
        help="Path to the trained diffusion model directory",
    )
    parser.add_argument(
        "--decoder_model_id",
        type=str,
        required=True,
        help="HuggingFace model ID for the latent decoder",
    )

    # Optional arguments
    parser.add_argument(
        "--num_samples",
        type=int,
        default=5,
        help="Number of samples to generate",
    )
    parser.add_argument(
        "--num_inference_steps",
        type=int,
        default=1000,
        help="Number of denoising steps",
    )
    parser.add_argument(
        "--max_new_tokens",
        type=int,
        default=100,
        help="Maximum number of new tokens to generate",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.7,
        help="Temperature for sampling",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed",
    )
    parser.add_argument(
        "--use_hf_unet",
        action="store_true",
        help="Whether the diffusion model uses HuggingFace UNet",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Device to run inference on",
    )

    return parser.parse_args()


def main():
    args = parse_args()

    # Set random seed
    torch.manual_seed(args.seed)
    generator = torch.Generator(device=args.device)
    generator.manual_seed(args.seed)

    logger.info(f"Generating text using latent diffusion model with parameters:")
    for arg, value in vars(args).items():
        logger.info(f"  {arg}: {value}")

    # Initialize pipeline
    logger.info(f"Initializing latent diffusion pipeline...")
    pipeline = LatentDiffusionPipeline(
        diffusion_model_path=args.diffusion_model_path,
        decoder_model_id=args.decoder_model_id,
        use_hf_unet=args.use_hf_unet,
        device=args.device,
    )

    # Generate text samples
    logger.info(f"Generating {args.num_samples} text samples...")
    samples = pipeline.generate(
        batch_size=args.num_samples,
        generator=generator,
        num_inference_steps=args.num_inference_steps,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
    )

    # Display generated samples
    logger.info("\n--- Generated Samples ---")
    for i, sample in enumerate(samples):
        logger.info(f"Sample {i+1}:")
        logger.info(f"{sample}")
        logger.info("------------------------")


if __name__ == "__main__":
    main()
