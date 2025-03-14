import torch
from latent_llm.models.gpt_latent import GPTLatentVAEPipeline
import argparse
import logging
from rich.logging import RichHandler

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(message)s", datefmt="[%X]", handlers=[RichHandler()]
)
logger = logging.getLogger(__name__)


def parse_args():
    parser = argparse.ArgumentParser(description="Inference with latent LLM")
    parser.add_argument(
        "--encoder_model_id",
        type=str,
        required=True,
        help="HuggingFace model ID for the encoder",
    )
    parser.add_argument(
        "--decoder_model_id",
        type=str,
        required=True,
        help="HuggingFace model ID for the decoder",
    )
    parser.add_argument(
        "--max_new_tokens",
        type=int,
        default=512,
        help="Maximum number of new tokens to generate",
    )
    parser.add_argument(
        "--temperature", type=float, default=0.7, help="Sampling temperature"
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument(
        "--input_text",
        type=str,
        default=None,
        help="Input text (if not provided, a random text will be used)",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    # Set random seed
    torch.manual_seed(args.seed)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info(f"Using device: {device}")

    # Initialize the pipeline
    logger.info(
        f"Initializing pipeline with encoder: {args.encoder_model_id}, decoder: {args.decoder_model_id}"
    )
    pipeline = GPTLatentVAEPipeline(
        pretrained_encoder_id=args.encoder_model_id,
        pretrained_decoder_id=args.decoder_model_id,
        torch_dtype=torch.bfloat16,
    )

    # Prepare input text
    if args.input_text:
        input_text = args.input_text
        logger.info(f"Using provided input text: {input_text[:50]}...")
    else:
        # Generate random text if none provided
        input_text = "This is a sample text to demonstrate the latent LLM encoding and decoding process."
        logger.info(f"Using default input text: {input_text}")

    # Encode input text to latent representation
    logger.info("Encoding input text to latent representation...")
    latent_embeds, _ = pipeline.encode(input_text)

    print(latent_embeds.mean(), latent_embeds.std())

    # Decode from latent representation
    logger.info("Generating text from latent representation...")
    generated_text = pipeline.decode(
        latent_embeds,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
    )

    logger.info("\n--- Results ---")
    logger.info(f"Input text: {input_text}")
    logger.info(f"Generated text: {generated_text}")
    logger.info("---------------")


if __name__ == "__main__":
    main()
