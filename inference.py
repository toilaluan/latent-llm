import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from latent_llm.models.gpt_latent import LatentEncoder, LatentDecoder
import argparse
import logging
from rich.logging import RichHandler
import numpy as np
from huggingface_hub import HfApi

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
        "--base_model_id",
        type=str,
        default="HuggingFaceTB/SmolLM2-135M",
        help="Base model ID",
    )
    parser.add_argument(
        "--n_gist_tokens", type=int, default=256, help="Number of gist tokens"
    )
    parser.add_argument(
        "--n_ae_tokens", type=int, default=1, help="Number of autoencoder tokens"
    )
    parser.add_argument("--block_size", type=int, default=256, help="Block size")
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

    # Load tokenizer
    logger.info(f"Loading tokenizer from {args.encoder_model_id}")
    tokenizer = AutoTokenizer.from_pretrained(args.encoder_model_id)

    # Load encoder and decoder
    logger.info(f"Loading encoder from {args.encoder_model_id}")
    try:
        # Try loading using the from_pretrained class method
        encoder = LatentEncoder.from_pretrained(args.encoder_model_id)
        encoder.to(device)
    except Exception as e:
        logger.error(f"Error loading encoder with from_pretrained: {e}")
        # Fall back to manual initialization and loading
        encoder = LatentEncoder(
            args.base_model_id, args.n_gist_tokens, args.n_ae_tokens, args.block_size
        )

    encoder.eval()

    logger.info(f"Loading decoder from {args.decoder_model_id}")
    # For decoder, we don't have a load_pretrained method, so we load the base model directly
    decoder = LatentDecoder(
        args.base_model_id, args.n_gist_tokens, args.n_ae_tokens, args.block_size
    )
    # The decoder just needs to load the model, no special tokens to load
    decoder.to(device)
    decoder.eval()

    # Prepare input text
    if args.input_text:
        input_text = args.input_text
        logger.info(f"Using provided input text: {input_text[:50]}...")
    else:
        # Generate random text if none provided
        input_text = "This is a sample text to demonstrate the latent LLM encoding and decoding process."
        logger.info(f"Using default input text: {input_text}")

    # Tokenize input
    input_tokens = tokenizer(
        input_text,
        truncation=True,
        max_length=args.block_size,
        padding="max_length",
        return_tensors="pt",
    )
    input_ids = input_tokens.input_ids.to(device)

    logger.info("Encoding input text to latent representation...")
    with torch.no_grad():
        # Encode the input text to latent representation
        latent_embeds = encoder(input_ids, pad_token_id=tokenizer.pad_token_id)

        # Decode from latent representation
        logger.info("Generating text from latent representation...")
        # Create empty input_ids (don't use first token as seed)
        logger.info(f"Latent embeds: {latent_embeds.shape}")

        generated_ids = decoder.generate(
            embeds=latent_embeds,
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
        )

    # Decode the generated tokens
    generated_text = tokenizer.decode(generated_ids[0], skip_special_tokens=True)

    logger.info("\n--- Results ---")
    logger.info(f"Input text: {input_text}")
    logger.info(f"Generated text: {generated_text}")
    logger.info("---------------")


if __name__ == "__main__":
    main()
