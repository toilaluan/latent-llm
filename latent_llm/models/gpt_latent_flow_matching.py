from latent_llm.get_tokenizer import get_tokenizer
from transformers import (
    AutoModelForCausalLM,
    AutoModelForMaskedLM,
    AutoConfig,
    AutoModel,
)
import torch
import torch.nn as nn
import torch.nn.functional as F
import os
import math
import numpy as np
from huggingface_hub import snapshot_download
from peft import LoraConfig, get_peft_model, TaskType, PeftModel


class GPTLatentFlowMatching(nn.Module):
    def __init__(
        self,
        model_name: str,
        latent_size: int,
        block_size: int,
        max_steps: int = 1000,
        timestep_token_size: int = 4,
        torch_dtype: torch.dtype = torch.bfloat16,
        device: str = "cuda",
        hidden_size: int = 896,
    ):
        super().__init__()
        self.model_name = model_name
        self.latent_size = latent_size
        self.block_size = block_size
        self.max_steps = max_steps
        self.timestep_token_size = timestep_token_size
        self.torch_dtype = torch_dtype
        self.device = device
        self.tokenizer = get_tokenizer(model_name)
        self.base_config = AutoConfig.from_pretrained(model_name)
        self.base_config.hidden_size = hidden_size
        self.model = AutoModel.from_config(self.base_config).to(dtype=torch_dtype)
        self.base_config = self.model.config

        self.timestep_proj = nn.Sequential(
            nn.Linear(1, self.base_config.hidden_size),
            nn.SiLU(),
            nn.Linear(
                self.base_config.hidden_size,
                self.base_config.hidden_size,
            ),
        )

        self.latent_shape = (self.latent_size, self.base_config.hidden_size)

        # Add learnable x1 parameter
        self.x1 = nn.Parameter(
            torch.randn(self.latent_size, self.base_config.hidden_size),
            requires_grad=True,
        )

        # Initialize all model weights
        self.initialize_weights()

    def initialize_weights(self):
        """Initialize model weights following best practices for flow matching models"""
        # Initialize x1 learnable parameter
        nn.init.normal_(self.x1, mean=0.0, std=0.02)

        # Initialize timestep embedding modules with Kaiming initialization
        for m in self.timestep_proj.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, a=0, mode="fan_in")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

        # Initialize transformer components if not pretrained
        # Note: For many pretrained models, this will be overridden by loaded weights
        for module in self.model.modules():
            if isinstance(module, nn.Linear):
                # Use slightly smaller init for stability
                nn.init.normal_(module.weight, mean=0.0, std=0.02)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.LayerNorm):
                nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)

    def get_timestep_tokens(self, timesteps: list[int]) -> torch.Tensor:
        """
        Get timestep embeddings using a simple MLP.

        Args:
            timesteps: List of timesteps, one per batch item

        Returns:
            Tensor of shape [B, timestep_token_size, D] containing timestep embeddings
        """
        # Convert timesteps to tensor and normalize to [0, 1] range
        timesteps_tensor = (
            torch.tensor(timesteps, device=self.device) / self.max_steps
        ).to(dtype=self.torch_dtype)

        # Get embeddings through the timestep embedding module
        t_embs = self.timestep_proj(timesteps_tensor)

        # Reshape to [B, 1, D]
        return t_embs.view(-1, 1, self.base_config.hidden_size)

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        latents: torch.Tensor,
        timesteps: list[int],
    ) -> torch.Tensor:
        """
        Forward pass for the flow matching model.

        Args:
            input_ids: Input token IDs [B, S]
            attention_mask: Mask for input tokens [B, S]
            latents: Latent representation to predict vector field for [B, T, D]
            timesteps: Timesteps for the flow process, one per batch item

        Returns:
            Predicted vector field for the latents
        """
        assert (
            attention_mask.shape[1] == input_ids.shape[1]
        ), "Attention mask shape mismatch"
        assert latents.shape[0] == input_ids.shape[0], "Batch size mismatch"
        B, T, D = latents.shape
        assert self.latent_shape == (T, D), "Latent shape mismatch"
        latents = latents.to(self.device)

        t_embs = self.get_timestep_tokens(timesteps)
        t_embs = t_embs.to(self.device)

        embeds = self.model.get_input_embeddings()(input_ids)
        embeds = torch.cat([embeds, t_embs, latents], dim=1)
        attention_mask = torch.cat(
            [
                attention_mask,
                torch.ones(B, t_embs.size(1), device=self.device),
                torch.ones(B, latents.size(1), device=self.device),
            ],
            dim=1,
        )
        output = self.model(
            inputs_embeds=embeds,
            attention_mask=attention_mask,
            output_hidden_states=True,
        )
        return output.last_hidden_state[:, -T:, :]

    def get_noised_latent(
        self, latents: torch.Tensor, timesteps: list[int]
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Apply noise to latent vectors based on timesteps.

        Args:
            latents: Original latent vectors [B, T, D]
            timesteps: List of timesteps, one per batch item

        Returns:
            Tuple of (noised_latents, vector_field_target)
        """
        assert all(t <= self.max_steps for t in timesteps), "Timestep out of range"
        B = latents.size(0)
        sigmas = torch.tensor(
            [[t / self.max_steps] for t in timesteps], device=self.device
        )  # Shape [B, 1]
        sigmas = sigmas.to(self.torch_dtype)
        # Reshape sigmas to [B, 1, 1] for proper broadcasting with [B, T, D]
        sigmas = sigmas.unsqueeze(-1)

        # Use learned x1 as target instead of random noise
        z1 = (
            self.x1.unsqueeze(0)
            .expand(B, -1, -1)
            .to(device=self.device, dtype=self.torch_dtype)
        )

        # Interpolate between source and learned target
        noised_latents = (1.0 - sigmas) * latents + sigmas * z1

        # Target vector field now points toward learned x1
        vector_field = z1 - latents

        return noised_latents, vector_field

    def get_loss(
        self, input_ids: torch.Tensor, latents: torch.Tensor, timesteps: list[int]
    ) -> torch.Tensor:
        """
        Calculate flow matching loss.

        Args:
            input_ids: Input token IDs [B, S]
            latents: Original latent vectors [B, T, D]
            timesteps: List of timesteps, one per batch item

        Returns:
            Flow matching loss
        """
        # Create attention mask
        attention_mask = (input_ids != self.tokenizer.pad_token_id).long()
        # Get noised latents and target vector field
        noised_latents, target_vector_field = self.get_noised_latent(latents, timesteps)
        # Predict vector field
        predicted_vector_field = self.forward(
            input_ids=input_ids,
            attention_mask=attention_mask,
            latents=noised_latents,
            timesteps=timesteps,
        )

        return torch.mean((predicted_vector_field - target_vector_field) ** 2)

    def sample(
        self,
        input_ids: torch.Tensor,
        num_steps: int = 100,
    ) -> torch.Tensor:
        """
        Sample from the model using numerical integration.

        Args:
            input_ids: Input token IDs [B, S]
            num_steps: Number of steps for the integration

        Returns:
            Sampled latent vectors
        """
        B = input_ids.shape[0]
        initial_noise = self.x1.unsqueeze(0).expand(B, -1, -1)
        initial_noise = initial_noise.to(self.device, dtype=self.torch_dtype)
        input_ids = input_ids.to(self.device)
        attention_mask = (input_ids != self.tokenizer.pad_token_id).long()
        current_latent = initial_noise

        timesteps = torch.linspace(
            self.max_steps, 1, num_steps, device=self.device
        ).int()

        # Calculate step sizes (may vary with non-linear schedules)
        step_sizes = []
        for i in range(len(timesteps) - 1):
            step_sizes.append(
                (timesteps[i] - timesteps[i + 1]).float() / self.max_steps
            )
        step_sizes.append(timesteps[-1].float() / self.max_steps)  # Last step
        print("Timesteps: ", timesteps)
        # Integration loop
        for i, step in enumerate(timesteps):
            # Current timestep for all items in batch
            batch_timesteps = [step.item()] * input_ids.shape[0]

            # Get vector field at current point
            with torch.no_grad():
                vector_field = self.forward(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    latents=current_latent,
                    timesteps=batch_timesteps,
                )
            current_latent = current_latent - step_sizes[i] * vector_field

            print(
                f"step {step}: {current_latent.mean().item():.4f}, {current_latent.std().item():.4f}"
            )

        return current_latent

    @classmethod
    def from_pretrained(
        cls,
        model_path: str,
        base_model_name: str,
        latent_size: int,
        block_size: int,
        torch_dtype: torch.dtype = torch.bfloat16,
        device: str = "cuda",
        use_lora: bool = True,
    ):
        """
        Load a pretrained model with LoRA weights.

        Args:
            model_path: Path to the saved model weights
            base_model_name: Name of the base model
            n_gist_tokens: Number of gist tokens
            block_size: Block size for the model
            torch_dtype: Data type for model parameters
            device: Device to load the model on
            use_lora: Whether the saved model uses LoRA

        Returns:
            Loaded model instance
        """
        # Initialize with base model
        model = cls(
            model_name=base_model_name,
            latent_size=latent_size,
            block_size=block_size,
            torch_dtype=torch_dtype,
            device=device,
            use_lora=use_lora,
        )

        # Load the saved weights
        if use_lora:
            # For LoRA models, we load the base model first, then the adapter
            if os.path.isdir(model_path):
                # Load from local path
                model.model = PeftModel.from_pretrained(
                    model.model, model_path, is_trainable=True
                )
            else:
                # Load from HuggingFace Hub
                model.model = PeftModel.from_pretrained(
                    model.model, model_path, is_trainable=True
                )
        else:
            # For full models, load the full state dict
            model.load_state_dict(torch.load(model_path, map_location=device))

        model.to(device)
        return model

    def save_pretrained(self, save_path: str):
        """
        Save the model weights.

        Args:
            save_path: Path to save the model to
        """
        os.makedirs(save_path, exist_ok=True)

        if self.use_lora:
            # Save only the LoRA weights
            self.model.save_pretrained(save_path)
        else:
            # Save the full model
            torch.save(self.state_dict(), os.path.join(save_path, "model.pt"))

        # Save the config
        import json

        config = {
            "model_name": self.model_name,
            "latent_size": self.latent_size,
            "block_size": self.block_size,
            "max_steps": self.max_steps,
            "timestep_token_size": self.timestep_token_size,
            "use_lora": self.use_lora,
        }
        with open(os.path.join(save_path, "config.json"), "w") as f:
            json.dump(config, f)


class GPTLatentFlowMatchingPipeline:
    def __init__(
        self,
        encoder_model_id: str,
        flow_matching_model_id: str,
        decoder_model_id: str = None,
        torch_dtype: torch.dtype = torch.bfloat16,
        device: str = "cuda",
        use_lora: bool = False,
    ):
        """
        Pipeline for text completion using flow matching.

        Args:
            encoder_model_id: HuggingFace repo ID for the encoder model
            flow_matching_model_id: HuggingFace repo ID for the flow matching model
            decoder_model_id: HuggingFace repo ID for the decoder model
            torch_dtype: Data type for model parameters
            device: Device to run models on
            use_lora: Whether the flow matching model uses LoRA
        """
        from latent_llm.models.gpt_latent import LatentEncoder, LatentDecoder
        import json

        self.device = device
        self.torch_dtype = torch_dtype

        # Load encoder
        self.encoder = LatentEncoder.from_pretrained(
            encoder_model_id, torch_dtype=torch_dtype, device=device
        )
        self.encoder.eval()

        # Load tokenizer
        self.tokenizer = self.encoder.tokenizer

        # Extract model parameters
        try:
            config_path = snapshot_download(
                repo_id=encoder_model_id,
                allow_patterns=["latent_config.json"],
            )
            config_path = os.path.join(config_path, "latent_config.json")

            with open(config_path, "r") as f:
                self.config = json.load(f)
        except Exception as e:
            print(f"Error loading config: {e}")
            raise

        # Load flow matching model
        if os.path.isdir(flow_matching_model_id):
            # Local path
            model_path = os.path.join(flow_matching_model_id, "model.pt")
            config_path = os.path.join(flow_matching_model_id, "config.json")

            with open(config_path, "r") as f:
                flow_config = json.load(f)

            # Check if this is a LoRA model
            if use_lora or (flow_config.get("use_lora", False)):
                self.flow_model = GPTLatentFlowMatching.from_pretrained(
                    model_path=flow_matching_model_id,
                    base_model_name=flow_config["model_name"],
                    latent_size=self.config["latent_size"],
                    block_size=self.config["block_size"],
                    torch_dtype=torch_dtype,
                    device=device,
                    use_lora=True,
                )
            else:
                self.flow_model = GPTLatentFlowMatching(
                    model_name=flow_config["model_name"],
                    latent_size=self.config["latent_size"],
                    block_size=self.config["block_size"],
                    torch_dtype=torch_dtype,
                    device=device,
                )
                # Load state dict
                self.flow_model.load_state_dict(
                    torch.load(model_path, map_location=device)
                )
        else:
            # HuggingFace Hub ID
            # Download config
            flow_config_path = snapshot_download(
                repo_id=flow_matching_model_id,
                allow_patterns=["config.json", "model.pt", "adapter_config.json"],
            )

            # Load config
            with open(os.path.join(flow_config_path, "config.json"), "r") as f:
                flow_config = json.load(f)

            # Check if this is a LoRA model
            if use_lora or os.path.exists(
                os.path.join(flow_config_path, "adapter_config.json")
            ):
                self.flow_model = GPTLatentFlowMatching.from_pretrained(
                    model_path=flow_matching_model_id,
                    base_model_name=flow_config["model_name"],
                    latent_size=self.config["latent_size"],
                    block_size=self.config["block_size"],
                    torch_dtype=torch_dtype,
                    device=device,
                    use_lora=True,
                )
            else:
                # Initialize model
                self.flow_model = GPTLatentFlowMatching(
                    model_name=flow_config["model_name"],
                    latent_size=self.config["latent_size"],
                    block_size=self.config["block_size"],
                    torch_dtype=torch_dtype,
                    device=device,
                )

                # Load weights
                self.flow_model.load_state_dict(
                    torch.load(
                        os.path.join(flow_config_path, "model.pt"), map_location=device
                    )
                )

        self.flow_model.to(device)
        self.flow_model.eval()

        # Load decoder
        if decoder_model_id is None:
            decoder_model_id = encoder_model_id

        # Check if decoder is same as encoder
        if decoder_model_id == encoder_model_id:
            # Create decoder from same base model
            self.decoder = LatentDecoder(
                model_name=decoder_model_id,
                latent_size=self.config["latent_size"],
                block_size=self.config["block_size"],
                torch_dtype=torch_dtype,
            )
        else:
            # Load separate decoder
            self.decoder = LatentDecoder(
                model_name=decoder_model_id,
                latent_size=self.config["latent_size"],
                block_size=self.config["block_size"],
                torch_dtype=torch_dtype,
            )

        self.decoder.to(device)
        self.decoder.eval()

    def encode_prefix(self, text: str) -> torch.Tensor:
        """Encode prefix text to token IDs"""
        inputs = self.tokenizer(
            text,
            return_tensors="pt",
            padding="max_length",
            max_length=self.encoder.block_size,
            truncation=True,
        )
        input_ids = inputs.input_ids.to(self.device)
        return input_ids

    def complete_text(
        self,
        prefix_text: str,
        num_steps: int = 100,
        max_new_tokens: int = 100,
        temperature: float = 0.7,
        num_samples: int = 1,
        return_all_samples: bool = False,
    ) -> str:
        """
        Complete text using flow matching

        Args:
            prefix_text: Text to complete
            num_steps: Number of steps for flow matching sampling
            max_new_tokens: Maximum number of tokens to generate
            temperature: Temperature for decoding
            num_samples: Number of completions to generate
            return_all_samples: Whether to return all samples or just the first one

        Returns:
            Completed text(s)
        """
        # Encode prefix text
        prefix_tokens = self.encode_prefix(prefix_text)

        # Generate samples
        if num_samples == 1 and not return_all_samples:
            # Single sample case
            return self._generate_single_completion(
                prefix_tokens, num_steps, max_new_tokens, temperature
            )
        else:
            # Multiple samples case
            completions = []
            for _ in range(num_samples):
                completion = self._generate_single_completion(
                    prefix_tokens, num_steps, max_new_tokens, temperature
                )
                completions.append(completion)

            if return_all_samples:
                return completions
            else:
                # Return first sample by default
                return completions[0]

    def _generate_single_completion(
        self,
        prefix_tokens: torch.Tensor,
        num_steps: int,
        max_new_tokens: int,
        temperature: float,
    ) -> str:
        """Generate a single completion"""
        # Generate initial noise
        B, S = prefix_tokens.shape
        initial_noise = torch.randn(
            B,
            self.flow_model.latent_size,
            self.flow_model.base_config.hidden_size,
            device=self.device,
        )

        # Sample using flow matching
        with torch.no_grad():
            # Generate latent representation for suffix
            predicted_latents = self.flow_model.sample(
                input_ids=prefix_tokens,
                initial_noise=initial_noise,
                num_steps=num_steps,
            )

            # Decode latents to text
            output_ids = self.decoder.generate(
                predicted_latents,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
            )

        # Decode output tokens
        output_text = self.tokenizer.decode(output_ids[0], skip_special_tokens=True)

        # Return full completion (prefix + generated)
        prefix_text = self.tokenizer.decode(
            prefix_tokens[0][prefix_tokens[0] != self.tokenizer.pad_token_id],
            skip_special_tokens=True,
        )

        return output_text


if __name__ == "__main__":
    # Example usage
    encoder_id = "your-encoder-model-id"
    flow_model_id = "your-flow-model-id"
    decoder_id = "your-decoder-model-id"  # Optional, defaults to encoder

    pipeline = GPTLatentFlowMatchingPipeline(
        encoder_model_id=encoder_id,
        flow_matching_model_id=flow_model_id,
        decoder_model_id=decoder_id,
    )

    prefix = "Once upon a time, there was a"

    # Generate a single completion
    completion = pipeline.complete_text(
        prefix_text=prefix, num_steps=100, max_new_tokens=50, temperature=0.7
    )

    print(f"Prefix: {prefix}")
    print(f"Completion: {completion}")

    # Generate multiple completions
    completions = pipeline.complete_text(
        prefix_text=prefix, num_samples=3, return_all_samples=True
    )

    print("\nMultiple completions:")
    for i, comp in enumerate(completions):
        print(f"{i+1}: {comp}")
