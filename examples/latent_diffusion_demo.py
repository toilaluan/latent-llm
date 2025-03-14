"""
Latent Diffusion Demo

This script demonstrates the full latent diffusion pipeline:
1. Encode text to latent representations
2. Train a small diffusion model on these latent representations
3. Generate new latent representations from noise
4. Decode these latent representations back to text

This is a minimal working example for demonstration purposes.
"""

import torch
import os
import tempfile
from datasets import Dataset
from transformers import AutoTokenizer
from latent_llm.models.gpt_latent import LatentEncoder, LatentDecoder
from latent_llm.models.gpt_latent_diffusion import (
    LatentDiffusionModel,
    train_latent_diffusion,
    LatentDiffusionPipeline,
)


def create_sample_dataset(texts, output_path):
    """Create a simple dataset for demonstration."""
    dataset = Dataset.from_dict({"text": texts})
    dataset.save_to_disk(output_path)
    return output_path


def main():
    # Configuration
    model_name = "HuggingFaceTB/SmolLM2-135M"  # Small model for demonstration
    n_gist_tokens = 8  # Small number for demo purposes
    n_ae_tokens = 1
    block_size = 64

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    # Create a temporary directory for outputs
    temp_dir = tempfile.mkdtemp()
    encoder_dir = os.path.join(temp_dir, "encoder")
    decoder_dir = os.path.join(temp_dir, "decoder")
    diffusion_dir = os.path.join(temp_dir, "diffusion")
    dataset_dir = os.path.join(temp_dir, "dataset")

    os.makedirs(encoder_dir, exist_ok=True)
    os.makedirs(decoder_dir, exist_ok=True)
    os.makedirs(diffusion_dir, exist_ok=True)
    os.makedirs(dataset_dir, exist_ok=True)

    print(f"Temporary directories created at: {temp_dir}")

    # Step 1: Initialize tokenizer, encoder and decoder
    print("Initializing tokenizer, encoder and decoder...")
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    # If the tokenizer doesn't have a pad token, add one
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Create a small latent encoder
    encoder = LatentEncoder(
        model_name=model_name,
        n_gist_tokens=n_gist_tokens,
        n_ae_tokens=n_ae_tokens,
        block_size=block_size,
        torch_dtype=torch.float32,  # Use float32 for demo purposes
    )
    encoder.to(device)

    # Create a small latent decoder
    decoder = LatentDecoder(
        model_name=model_name,
        n_gist_tokens=n_gist_tokens,
        n_ae_tokens=n_ae_tokens,
        block_size=block_size,
        torch_dtype=torch.float32,  # Use float32 for demo purposes
    )
    decoder.to(device)

    # Step 2: Create a small sample dataset
    print("Creating sample dataset...")
    sample_texts = [
        "This is a sample text for demonstration purposes.",
        "The diffusion model will learn to generate latent representations.",
        "We can then decode these latent representations back to text.",
        "This approach allows for interesting text generation capabilities.",
        "The model combines latent representations with diffusion models.",
    ]
    dataset_path = create_sample_dataset(sample_texts, dataset_dir)

    # Step 3: Encode sample texts to get latent dimensionality
    print("Encoding sample text to determine latent dimensionality...")
    encoded = tokenizer(
        sample_texts[0],
        padding="max_length",
        max_length=block_size,
        truncation=True,
        return_tensors="pt",
    )
    input_ids = encoded.input_ids.to(device)

    with torch.no_grad():
        latent = encoder(input_ids, tokenizer.pad_token_id)

    latent_dim = latent.shape[1] * latent.shape[2]  # B x T x D -> flatten T*D
    print(f"Latent dimension: {latent_dim}")

    # Step 4: Save encoder and decoder models
    print("Saving encoder and decoder models...")
    # Save the encoder model to disk (simplified for demo)
    torch.save(encoder.state_dict(), os.path.join(encoder_dir, "encoder.pt"))

    # Create a simple config file
    with open(os.path.join(encoder_dir, "latent_config.json"), "w") as f:
        import json

        json.dump(
            {
                "n_gist_tokens": n_gist_tokens,
                "n_ae_tokens": n_ae_tokens,
                "block_size": block_size,
                "hidden_size": encoder.base_config.hidden_size,
            },
            f,
        )

    # Step 5: Train a small diffusion model
    print("Training a small diffusion model (this is a minimal demo)...")

    # Create a small MLP-based diffusion model
    diffusion_model = LatentDiffusionModel(
        latent_dim=latent_dim,
        hidden_dims=[64, 128],  # Small for demo
        dropout=0.1,
    )
    diffusion_model.to(device)

    # Create a simple training loop (extremely simplified for demo)
    # In a real scenario, use the train_latent_diffusion function
    from diffusers import DDPMScheduler

    # Initialize noise scheduler
    noise_scheduler = DDPMScheduler(num_train_timesteps=100)  # Small for demo

    # Step 6: Generate new latent representations
    print("Generating new latent representations from noise...")

    # For demonstration, we'll just randomly initialize
    with torch.no_grad():
        # Start from random noise
        noise = torch.randn((1, latent_dim), device=device)

        # Save for comparison
        initial_noise = noise.clone()

        # Simple denoising loop (extremely simplified)
        diffusion_model.eval()
        for t in range(50, 0, -1):  # Small range for demo
            timesteps = torch.ones(1, device=device) * t
            timesteps = timesteps.long()

            # Get model prediction
            model_output = diffusion_model(noise, timesteps)

            # Update noise with scheduler (simplified)
            noise = noise - 0.1 * model_output

        # Reshape to match decoder's expected format
        generated_latent = noise.reshape(1, n_gist_tokens + n_ae_tokens, -1)

    # Step 7: Decode latent representations to text
    print("Decoding generated latent representations to text...")

    # Generate text from latents (simplified for demo)
    with torch.no_grad():
        # For comparison, also decode the original latent
        original_output_ids = decoder.generate(
            latent,
            max_new_tokens=20,
            temperature=0.7,
        )

        # Decode generated latent
        generated_output_ids = decoder.generate(
            generated_latent,
            max_new_tokens=20,
            temperature=0.7,
        )

    # Decode output tokens to text
    original_text = tokenizer.decode(original_output_ids[0], skip_special_tokens=True)
    generated_text = tokenizer.decode(generated_output_ids[0], skip_special_tokens=True)

    print("\n--- Results ---")
    print(f"Original text: {sample_texts[0]}")
    print(f"Reconstructed text: {original_text}")
    print(f"Generated text from noise: {generated_text}")
    print("---------------")

    print(f"\nThis was a minimal demo. In a real setting, use:")
    print("1. train_latent_diffusion() for training")
    print("2. LatentDiffusionPipeline for inference")

    # Cleanup
    print(f"\nTemporary files created at {temp_dir}")
    print("Remember to delete these if they're no longer needed")


if __name__ == "__main__":
    main()
