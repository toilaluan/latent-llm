from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
import torch.nn as nn
import torch.nn.functional as F
import os
from huggingface_hub import snapshot_download


class GPTLatentFlowMatching(nn.Module):
    def __init__(
        self,
        model_name: str,
        n_gist_tokens: int,
        block_size: int,
        max_steps: int = 1000,
        timestep_token_size: int = 4,
        torch_dtype: torch.dtype = torch.bfloat16,
        device: str = "cuda",
    ):
        super().__init__()
        self.model_name = model_name
        self.n_gist_tokens = n_gist_tokens
        self.block_size = block_size
        self.max_steps = max_steps
        self.timestep_token_size = timestep_token_size
        self.torch_dtype = torch_dtype
        self.device = device
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.tokenizer.add_special_tokens({"pad_token": "<|pad|>"})
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name, torch_dtype=torch_dtype
        )
        self.base_config = self.model.config
        self.timestep_embeddings = nn.Embedding(
            num_embeddings=max_steps * timestep_token_size,
            embedding_dim=self.base_config.hidden_size,
            dtype=torch_dtype,
        )
        torch.nn.init.kaiming_normal_(self.timestep_embeddings.weight)
        self.latent_shape = (self.n_gist_tokens, self.base_config.hidden_size)

    def get_timestep_tokens(self, timesteps: list[int]) -> torch.Tensor:
        # if timestep is 1, return self.timestep_embeddings([0,1,..,timestep_token_size-1])
        # Return shape is [B, T, D]
        timestep_tokens = []
        for timestep in timesteps:
            indexes = list(
                range(
                    (timestep - 1) * self.timestep_token_size,
                    timestep * self.timestep_token_size,
                )
            )
            timesteps = self.timestep_embeddings(
                torch.tensor(indexes, device=self.device).long().unsqueeze(0)
            )  # [1, T, D]
            timestep_tokens.append(timesteps)
        return torch.cat(timestep_tokens, dim=0)

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
        return output.hidden_states[-1][:, -T:, :]

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
        sigmas = torch.tensor(
            [[t / self.max_steps] for t in timesteps], device=self.device
        )  # Shape [B, 1]
        sigmas = sigmas.to(self.torch_dtype)
        # Sample noise
        noise = torch.randn_like(latents, device=self.device, dtype=self.torch_dtype)
        # Interpolate between source and target
        noised_latents = (1.0 - sigmas) * latents + sigmas * noise
        # Target vector field for flow matching
        # This represents the direction toward the target distribution
        vector_field = noise - latents

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

        # Flow matching loss is MSE between predicted and target vector fields
        return F.mse_loss(predicted_vector_field, target_vector_field)

    def sample(
        self, input_ids: torch.Tensor, initial_noise: torch.Tensor, num_steps: int = 100
    ) -> torch.Tensor:
        """
        Sample from the model using Euler integration.

        Args:
            input_ids: Input token IDs [B, S]
            initial_noise: Initial noise to start sampling from [B, T, D]
            num_steps: Number of steps for the integration

        Returns:
            Sampled latent vectors
        """
        initial_noise = initial_noise.to(self.device, dtype=self.torch_dtype)
        input_ids = input_ids.to(self.device)
        attention_mask = (input_ids != self.tokenizer.pad_token_id).long()

        # Start with random noise
        current_latent = initial_noise
        step_size = 1.0 / num_steps

        # Euler integration
        for step in range(num_steps, 0, -1):
            # Current timestep for all items in batch
            timesteps = [step] * input_ids.shape[0]

            # Get vector field at current point
            with torch.no_grad():
                vector_field = self.forward(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    latents=current_latent,
                    timesteps=timesteps,
                )

            # Update using Euler method
            current_latent = current_latent + step_size * vector_field

        return current_latent


class GPTLatentFlowMatchingPipeline:
    def __init__(
        self,
        encoder_model_id: str,
        flow_matching_model_id: str,
        decoder_model_id: str = None,
        torch_dtype: torch.dtype = torch.bfloat16,
        device: str = "cuda",
    ):
        """
        Pipeline for text completion using flow matching.

        Args:
            encoder_model_id: HuggingFace repo ID for the encoder model
            flow_matching_model_id: HuggingFace repo ID for the flow matching model
            decoder_model_id: HuggingFace repo ID for the decoder model
            torch_dtype: Data type for model parameters
            device: Device to run models on
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

            self.flow_model = GPTLatentFlowMatching(
                model_name=flow_config["model_name"],
                n_gist_tokens=self.config["n_gist_tokens"],
                block_size=self.config["block_size"],
                torch_dtype=torch_dtype,
                device=device,
            )
            # Load state dict
            self.flow_model.load_state_dict(torch.load(model_path, map_location=device))
        else:
            # HuggingFace Hub ID
            # Download config
            flow_config_path = snapshot_download(
                repo_id=flow_matching_model_id,
                allow_patterns=["config.json", "model.pt"],
            )

            # Load config
            with open(os.path.join(flow_config_path, "config.json"), "r") as f:
                flow_config = json.load(f)

            # Initialize model
            self.flow_model = GPTLatentFlowMatching(
                model_name=flow_config["model_name"],
                n_gist_tokens=self.config["n_gist_tokens"],
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
                n_gist_tokens=self.config["n_gist_tokens"],
                n_ae_tokens=self.config["n_ae_tokens"],
                block_size=self.config["block_size"],
                torch_dtype=torch_dtype,
            )
        else:
            # Load separate decoder
            self.decoder = LatentDecoder(
                model_name=decoder_model_id,
                n_gist_tokens=self.config["n_gist_tokens"],
                n_ae_tokens=self.config["n_ae_tokens"],
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
            self.flow_model.n_gist_tokens,
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
