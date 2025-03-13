import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Dict, Any
import os
import numpy as np
from diffusers import UNet2DModel, DDPMScheduler
from tqdm.auto import tqdm
from accelerate import Accelerator
from accelerate.logging import get_logger
from diffusers.optimization import get_cosine_schedule_with_warmup
from huggingface_hub import create_repo, upload_folder
from pathlib import Path

from .gpt_latent import LatentEncoder

logger = get_logger(__name__)


class LatentDiffusionModel(nn.Module):
    """
    Diffusion model for generating latent representations matching those from the encoder.
    Instead of working with images, this diffusion model works with latent embeddings.
    """

    def __init__(
        self,
        latent_dim: int,
        hidden_dims: list = [256, 512, 1024],
        dropout: float = 0.1,
    ):
        super().__init__()

        # Simple MLP-based UNet architecture for latent space diffusion
        layers = []

        # Encoder (downsample)
        in_dim = latent_dim
        for h_dim in hidden_dims:
            layers.append(
                nn.Sequential(
                    nn.Linear(in_dim, h_dim),
                    nn.LayerNorm(h_dim),
                    nn.Dropout(dropout),
                    nn.GELU(),
                )
            )
            in_dim = h_dim

        # Bottleneck with timestep embedding
        self.time_embed = nn.Sequential(
            nn.Linear(hidden_dims[-1], hidden_dims[-1]),
            nn.GELU(),
            nn.Linear(hidden_dims[-1], hidden_dims[-1]),
        )

        # Decoder (upsample)
        reversed_hidden_dims = list(reversed(hidden_dims))
        for i, h_dim in enumerate(reversed_hidden_dims[1:]):
            layers.append(
                nn.Sequential(
                    nn.Linear(reversed_hidden_dims[i], h_dim),
                    nn.LayerNorm(h_dim),
                    nn.Dropout(dropout),
                    nn.GELU(),
                )
            )
            in_dim = h_dim

        # Final layer
        layers.append(nn.Linear(reversed_hidden_dims[-1], latent_dim))

        self.model = nn.ModuleList(layers)
        self.latent_dim = latent_dim

    def _sinusoidal_embedding(self, timesteps, dim):
        # Standard sinusoidal embedding for diffusion timesteps
        half_dim = dim // 2
        embeddings = np.log(10000) / (half_dim - 1)
        embeddings = torch.exp(
            torch.arange(half_dim, device=timesteps.device) * -embeddings
        )
        embeddings = timesteps[:, None] * embeddings[None, :]
        embeddings = torch.cat([torch.sin(embeddings), torch.cos(embeddings)], dim=1)
        return embeddings

    def forward(self, x, timesteps):
        # Embed timesteps
        t_emb = self._sinusoidal_embedding(timesteps, self.model[0][0].in_features)
        t_emb = self.time_embed(t_emb)

        # Forward pass through encoder part
        h = x
        skip_connections = []
        for i in range(len(self.model) // 2):
            h = self.model[i](h)
            skip_connections.append(h)

        # Add timestep embedding
        h = h + t_emb

        # Forward pass through decoder part with skip connections
        for i in range(len(self.model) // 2, len(self.model) - 1):
            h = self.model[i](h)
            h = h + skip_connections[-(i - len(self.model) // 2 + 1)]

        # Final layer
        h = self.model[-1](h)

        return h


class HFStyleUNetLatentDiffusion(nn.Module):
    """
    Diffusion model for latent space using HuggingFace's UNet2DModel
    by reshaping latent vectors to 2D tensors.
    """

    def __init__(
        self,
        latent_dim: int,
        reshape_factor: int = 8,  # Factor to reshape 1D latent to 2D
        in_channels: int = 4,
        out_channels: int = 4,
        model_config: Optional[Dict[str, Any]] = None,
    ):
        super().__init__()

        # Calculate the reshape dimensions
        # We'll reshape the latent to a 2D tensor to use with UNet2DModel
        total_elements = latent_dim
        height = width = int(np.sqrt(total_elements // in_channels))

        # Adjust if not an exact fit
        while height * width * in_channels < total_elements:
            height += 1

        self.height = height
        self.width = width
        self.in_channels = in_channels
        self.latent_dim = latent_dim

        # Calculate padding needed
        self.pad_size = height * width * in_channels - latent_dim

        if model_config is None:
            model_config = {
                "sample_size": height,  # the target image resolution
                "in_channels": in_channels,  # input channels
                "out_channels": out_channels,  # output channels
                "layers_per_block": 2,  # layers per UNet block
                "block_out_channels": (128, 256, 512, 512),  # channels per block
                "down_block_types": (
                    "DownBlock2D",
                    "DownBlock2D",
                    "AttnDownBlock2D",
                    "DownBlock2D",
                ),
                "up_block_types": (
                    "UpBlock2D",
                    "AttnUpBlock2D",
                    "UpBlock2D",
                    "UpBlock2D",
                ),
            }

        self.unet = UNet2DModel(**model_config)

    def forward(self, x, timesteps):
        batch_size = x.shape[0]

        # Pad if necessary
        if self.pad_size > 0:
            padding = torch.zeros(
                batch_size, self.pad_size, device=x.device, dtype=x.dtype
            )
            x_padded = torch.cat([x, padding], dim=1)
        else:
            x_padded = x

        # Reshape to 2D for UNet
        x_2d = x_padded.reshape(batch_size, self.in_channels, self.height, self.width)

        # Forward through UNet
        output = self.unet(x_2d, timesteps).sample

        # Reshape back and remove padding
        output_flat = output.reshape(batch_size, -1)
        if self.pad_size > 0:
            output_flat = output_flat[:, : self.latent_dim]

        return output_flat

    @property
    def config(self):
        """Get model configuration for saving."""
        return {
            "latent_dim": self.latent_dim,
            "reshape_factor": 8,  # Default value
            "in_channels": self.in_channels,
            "out_channels": self.unet.config.out_channels,
            "model_config": self.unet.config,
        }


def train_latent_diffusion(
    encoder_model_id: str,
    output_dir: str,
    dataset_path: str,
    num_epochs: int = 100,
    batch_size: int = 16,
    learning_rate: float = 1e-4,
    gradient_accumulation_steps: int = 1,
    lr_warmup_steps: int = 500,
    save_image_epochs: int = 10,
    save_model_epochs: int = 50,
    mixed_precision: str = "fp16",  # or "bf16"
    use_hf_unet: bool = True,  # Whether to use HuggingFace UNet or custom MLP
    push_to_hub: bool = False,
    hub_model_id: Optional[str] = None,
    seed: int = 42,
):
    """
    Train a diffusion model to generate latent representations.

    Args:
        encoder_model_id: HuggingFace model ID for the latent encoder
        output_dir: Directory to save model checkpoints
        dataset_path: Path to dataset containing text samples
        num_epochs: Number of training epochs
        batch_size: Batch size for training
        learning_rate: Learning rate
        gradient_accumulation_steps: Number of steps for gradient accumulation
        lr_warmup_steps: Number of warmup steps for learning rate scheduler
        save_image_epochs: Save sample generated latents every N epochs
        save_model_epochs: Save model checkpoint every N epochs
        mixed_precision: Mixed precision type (fp16 or bf16)
        use_hf_unet: Whether to use HuggingFace UNet or custom MLP
        push_to_hub: Whether to push model to Hugging Face Hub
        hub_model_id: Model ID for pushing to Hub
        seed: Random seed
    """
    # Set seed for reproducibility
    torch.manual_seed(seed)

    # Initialize accelerator
    accelerator = Accelerator(
        mixed_precision=mixed_precision,
        gradient_accumulation_steps=gradient_accumulation_steps,
        log_with="tensorboard",
        project_dir=os.path.join(output_dir, "logs"),
    )

    if accelerator.is_main_process:
        if output_dir is not None:
            os.makedirs(output_dir, exist_ok=True)
        if push_to_hub:
            repo_id = create_repo(
                repo_id=hub_model_id or Path(output_dir).name, exist_ok=True
            ).repo_id
        accelerator.init_trackers("latent_diffusion")

    # Load encoder model to generate latent embeddings
    logger.info(f"Loading encoder model {encoder_model_id}")
    encoder = LatentEncoder.from_pretrained(encoder_model_id)
    encoder.eval()  # Set to eval mode as we're only using it to generate latents

    # Get latent dimensionality
    # We'll create a small sample to determine latent dim
    dummy_input = torch.ones(1, encoder.block_size).long()
    dummy_pad_id = 0  # Assuming 0 is pad token ID
    with torch.no_grad():
        latent = encoder(dummy_input, dummy_pad_id)
    latent_dim = latent.shape[1] * latent.shape[2]  # B x T x D -> flatten T*D
    logger.info(f"Latent dimension: {latent_dim}")

    # Initialize diffusion model
    if use_hf_unet:
        logger.info("Using HuggingFace UNet for diffusion model")
        model = HFStyleUNetLatentDiffusion(latent_dim=latent_dim)
    else:
        logger.info("Using custom MLP network for diffusion model")
        model = LatentDiffusionModel(latent_dim=latent_dim)

    # Initialize noise scheduler
    noise_scheduler = DDPMScheduler(num_train_timesteps=1000)

    # Initialize optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

    # Create sample dataset - simplified for this example
    # In a real implementation, you would load actual text data
    # and encode it with the latent encoder
    import datasets
    from torch.utils.data import DataLoader

    # Load dataset
    raw_dataset = datasets.load_dataset(dataset_path)

    # Define preprocessing function to encode text to latents
    def encode_text_to_latent(examples):
        text_inputs = examples["text"]
        encoded = encoder.tokenizer(
            text_inputs,
            padding="max_length",
            max_length=encoder.block_size,
            truncation=True,
            return_tensors="pt",
        )
        input_ids = encoded.input_ids

        # Encode to latent space
        with torch.no_grad():
            latents = encoder(input_ids, encoder.tokenizer.pad_token_id)

        # Flatten the latents
        latents = latents.reshape(latents.shape[0], -1)

        return {"latents": latents}

    # Process the dataset
    processed_dataset = raw_dataset.map(
        encode_text_to_latent,
        batched=True,
        batch_size=32,
        remove_columns=["text"],
    )

    # Create dataloader
    train_dataloader = DataLoader(
        processed_dataset, batch_size=batch_size, shuffle=True
    )

    # Create learning rate scheduler
    lr_scheduler = get_cosine_schedule_with_warmup(
        optimizer=optimizer,
        num_warmup_steps=lr_warmup_steps,
        num_training_steps=(len(train_dataloader) * num_epochs),
    )

    # Prepare everything with accelerator
    model, optimizer, train_dataloader, lr_scheduler = accelerator.prepare(
        model, optimizer, train_dataloader, lr_scheduler
    )

    # Train loop
    global_step = 0

    for epoch in range(num_epochs):
        progress_bar = tqdm(
            total=len(train_dataloader), disable=not accelerator.is_local_main_process
        )
        progress_bar.set_description(f"Epoch {epoch}")

        model.train()

        for step, batch in enumerate(train_dataloader):
            clean_latents = batch["latents"].to(accelerator.device)

            # Sample noise to add to the latents
            noise = torch.randn_like(clean_latents)

            # Sample a random timestep for each latent
            bs = clean_latents.shape[0]
            timesteps = torch.randint(
                0,
                noise_scheduler.config.num_train_timesteps,
                (bs,),
                device=clean_latents.device,
            ).long()

            # Add noise to the clean latents according to the noise magnitude at each timestep
            noisy_latents = noise_scheduler.add_noise(clean_latents, noise, timesteps)

            with accelerator.accumulate(model):
                # Predict the noise residual
                noise_pred = model(noisy_latents, timesteps)
                loss = F.mse_loss(noise_pred, noise)

                accelerator.backward(loss)

                if accelerator.sync_gradients:
                    accelerator.clip_grad_norm_(model.parameters(), 1.0)

                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()

            progress_bar.update(1)
            logs = {
                "loss": loss.detach().item(),
                "lr": lr_scheduler.get_last_lr()[0],
                "step": global_step,
            }
            progress_bar.set_postfix(**logs)
            accelerator.log(logs, step=global_step)
            global_step += 1

        # After each epoch, optionally generate and save some samples
        if accelerator.is_main_process:
            if (epoch + 1) % save_model_epochs == 0 or epoch == num_epochs - 1:
                # Save the model checkpoint
                accelerator.wait_for_everyone()
                unwrapped_model = accelerator.unwrap_model(model)

                model_path = os.path.join(output_dir, f"checkpoint-{epoch}")
                os.makedirs(model_path, exist_ok=True)

                # For UNet models
                if use_hf_unet:
                    unwrapped_model.unet.save_pretrained(
                        os.path.join(model_path, "unet")
                    )

                    # Save configuration
                    import json

                    with open(os.path.join(model_path, "config.json"), "w") as f:
                        json.dump(unwrapped_model.config, f)

                # Save the full model state
                torch.save(
                    unwrapped_model.state_dict(), os.path.join(model_path, "model.pt")
                )

                if push_to_hub:
                    upload_folder(
                        repo_id=repo_id,
                        folder_path=model_path,
                        commit_message=f"Epoch {epoch}",
                    )

            # Generate and save sample latents
            if (epoch + 1) % save_image_epochs == 0 or epoch == num_epochs - 1:
                # Generate sample latents
                eval_batch_size = 4
                noise = torch.randn(
                    (eval_batch_size, latent_dim), device=accelerator.device
                )

                # Save noise for comparison
                torch.save(noise.cpu(), os.path.join(output_dir, f"noise-{epoch}.pt"))

                # Noise -> Latent generation
                model.eval()

                with torch.no_grad():
                    # Initialize from random noise
                    latents = noise

                    # DDPM sampling
                    for t in tqdm(
                        reversed(range(0, noise_scheduler.config.num_train_timesteps)),
                        desc="Sampling",
                        total=noise_scheduler.config.num_train_timesteps,
                    ):
                        timesteps = (
                            torch.ones(eval_batch_size, device=accelerator.device) * t
                        )
                        timesteps = timesteps.long()

                        # Get model prediction
                        model_output = model(latents, timesteps)

                        # Update latents with scheduler
                        latents = noise_scheduler.step(
                            model_output, t, latents
                        ).prev_sample

                # Save generated latents
                torch.save(
                    latents.cpu(), os.path.join(output_dir, f"latents-{epoch}.pt")
                )

    # Save the final model
    accelerator.wait_for_everyone()
    unwrapped_model = accelerator.unwrap_model(model)

    final_model_path = os.path.join(output_dir, "final_model")
    os.makedirs(final_model_path, exist_ok=True)

    # For UNet models
    if use_hf_unet:
        unwrapped_model.unet.save_pretrained(os.path.join(final_model_path, "unet"))

        # Save configuration
        import json

        with open(os.path.join(final_model_path, "config.json"), "w") as f:
            json.dump(unwrapped_model.config, f)

    # Save the full model state
    torch.save(unwrapped_model.state_dict(), os.path.join(final_model_path, "model.pt"))

    # Save noise scheduler
    noise_scheduler.save_pretrained(os.path.join(final_model_path, "scheduler"))

    if push_to_hub:
        upload_folder(
            repo_id=repo_id,
            folder_path=final_model_path,
            commit_message="Final model",
        )

    return unwrapped_model


class LatentDiffusionPipeline:
    """
    Pipeline for generating latent embeddings using a trained diffusion model
    and decoding them to text using the latent decoder.
    """

    def __init__(
        self,
        diffusion_model_path: str,
        decoder_model_id: str,
        use_hf_unet: bool = True,
        device: str = "cuda",
    ):
        """
        Initialize the latent diffusion pipeline.

        Args:
            diffusion_model_path: Path to the trained diffusion model
            decoder_model_id: Hugging Face model ID for the latent decoder
            use_hf_unet: Whether the diffusion model uses HuggingFace UNet
            device: Device to run inference on
        """
        self.diffusion_model_path = diffusion_model_path
        self.decoder_model_id = decoder_model_id
        self.use_hf_unet = use_hf_unet
        self.device = device

        # Load config
        import json

        with open(os.path.join(diffusion_model_path, "config.json"), "r") as f:
            self.config = json.load(f)

        # Load scheduler
        self.scheduler = DDPMScheduler.from_pretrained(
            os.path.join(diffusion_model_path, "scheduler")
        )

        # Load model
        if use_hf_unet:
            self.model = HFStyleUNetLatentDiffusion(
                latent_dim=self.config["latent_dim"],
                in_channels=self.config["in_channels"],
                out_channels=self.config["out_channels"],
                model_config=self.config["model_config"],
            )
            self.model.unet = UNet2DModel.from_pretrained(
                os.path.join(diffusion_model_path, "unet")
            )
        else:
            # For MLP-based model
            self.model = LatentDiffusionModel(
                latent_dim=self.config["latent_dim"],
                hidden_dims=self.config.get("hidden_dims", [256, 512, 1024]),
            )
            # Load weights
            self.model.load_state_dict(
                torch.load(os.path.join(diffusion_model_path, "model.pt"))
            )

        self.model.to(device)
        self.model.eval()

        # Load decoder and tokenizer
        from .gpt_latent import LatentDecoder
        import transformers

        # Get latent config from decoder model
        self.tokenizer = transformers.AutoTokenizer.from_pretrained(decoder_model_id)

        # Download config to get parameters for LatentDecoder
        config_path = os.path.join(diffusion_model_path, "latent_config.json")
        with open(config_path, "r") as f:
            latent_config = json.load(f)

        self.decoder = LatentDecoder(
            model_name=decoder_model_id,
            n_gist_tokens=latent_config["n_gist_tokens"],
            n_ae_tokens=latent_config["n_ae_tokens"],
            block_size=latent_config["block_size"],
        )
        self.decoder.to(device)
        self.decoder.eval()

    @torch.no_grad()
    def generate_latent(
        self,
        batch_size: int = 1,
        generator: Optional[torch.Generator] = None,
        num_inference_steps: int = 1000,
    ):
        """
        Generate latent embeddings using the diffusion model.

        Args:
            batch_size: Number of latents to generate
            generator: Random number generator
            num_inference_steps: Number of denoising steps

        Returns:
            Generated latent embeddings
        """
        # Start from random noise
        latents = torch.randn(
            (batch_size, self.config["latent_dim"]),
            generator=generator,
            device=self.device,
        )

        # Denoising loop
        for t in tqdm(reversed(range(0, num_inference_steps)), desc="Sampling"):
            timesteps = torch.ones(batch_size, device=self.device) * t
            timesteps = timesteps.long()

            # Get model prediction
            model_output = self.model(latents, timesteps)

            # Update latents with scheduler
            latents = self.scheduler.step(model_output, t, latents).prev_sample

        return latents

    @torch.no_grad()
    def generate(
        self,
        batch_size: int = 1,
        generator: Optional[torch.Generator] = None,
        num_inference_steps: int = 1000,
        max_new_tokens: int = 100,
        temperature: float = 0.7,
    ):
        """
        Generate text by first generating latent embeddings and then decoding them.

        Args:
            batch_size: Number of samples to generate
            generator: Random number generator
            num_inference_steps: Number of denoising steps
            max_new_tokens: Maximum number of tokens to generate
            temperature: Temperature for sampling

        Returns:
            Generated text
        """
        # Generate latent embeddings
        latents = self.generate_latent(
            batch_size=batch_size,
            generator=generator,
            num_inference_steps=num_inference_steps,
        )

        # Reshape latents to match decoder's expected format
        n_gist_tokens = self.config["n_gist_tokens"]
        n_ae_tokens = self.config["n_ae_tokens"]
        hidden_size = self.config["hidden_size"]

        # Reshape to match the decoder's expected format
        reshaped_latents = latents.reshape(
            batch_size, n_gist_tokens + n_ae_tokens, hidden_size
        )

        # Generate text from latents
        output_ids = self.decoder.generate(
            reshaped_latents,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
        )

        # Decode output tokens to text
        output_text = [
            self.tokenizer.decode(ids, skip_special_tokens=True) for ids in output_ids
        ]

        return output_text
