from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    T5Model,
    T5ForConditionalGeneration,
)
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple
import os
from huggingface_hub import HfApi
from huggingface_hub import snapshot_download
from safetensors.torch import save_file, load_file
import logging

logger = logging.getLogger(__name__)


# Add this code to enable debug logging
def _enable_debug_logging():
    """Configure logging to show debug messages."""
    logging.basicConfig(
        level=logging.DEBUG,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )
    logger.setLevel(logging.DEBUG)


# Call this function at the beginning of your script or before using the models
# enable_debug_logging()  # Uncomment this line to enable debug logging

CKPT_DIR = ".training_cache/checkpoints"

if not os.path.exists(CKPT_DIR):
    os.makedirs(CKPT_DIR)


class LatentEncoder(nn.Module):
    def __init__(
        self,
        model_name: str,
        n_gist_tokens: int,
        block_size: int,
        torch_dtype: torch.dtype = torch.bfloat16,
        kl_weight: float = 1e-4,
        enable_debug_logging: bool = False,
    ):
        super().__init__()
        self.model = T5Model.from_pretrained(
            model_name,
            torch_dtype=torch_dtype,
        )
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.base_config = self.model.config

        # For VAE, we need separate parameters for mean and log variance
        self.gist_tokens_mean = nn.Parameter(
            torch.randn(n_gist_tokens, self.base_config.d_model, dtype=torch_dtype)
        )
        self.gist_tokens_logvar = nn.Parameter(
            torch.zeros(n_gist_tokens, self.base_config.d_model, dtype=torch_dtype)
        )
        self.n_gist_tokens = n_gist_tokens
        self.block_size = block_size
        self.latent_config = {
            "n_gist_tokens": n_gist_tokens,
            "block_size": block_size,
        }
        self.kl_weight = kl_weight
        self.init_weights()
        self.enable_debug_logging = enable_debug_logging
        if enable_debug_logging:
            _enable_debug_logging()

    def init_weights(self):
        torch.nn.init.kaiming_normal_(self.gist_tokens_mean)
        torch.nn.init.zeros_(self.gist_tokens_logvar)

    def reparameterize(self, mean, logvar):
        """
        Reparameterization trick: sample from a standard normal and scale by
        standard deviation and shift by mean for backpropagation.
        """
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mean + eps * std

    def kl_divergence(self, mean, logvar):
        """
        Compute KL divergence between the latent distribution and standard normal
        Input shapes: (batch_size, n_gist_tokens, hidden_size)
        Output shape: scalar (averaged over batch, summed over latent dimensions)
        """
        # Reshape to combine n_gist_tokens and hidden_size dimensions for proper summing
        batch_size = mean.size(0)
        reshaped_mean = mean.reshape(batch_size, -1)
        reshaped_logvar = logvar.reshape(batch_size, -1)

        # Calculate KL divergence components
        kl_per_batch = (
            1 + reshaped_logvar - reshaped_mean.pow(2) - reshaped_logvar.exp()
        )
        # Sum over latent dimensions and average over batch
        return -0.5 * torch.mean(torch.sum(kl_per_batch, dim=1))

    def forward(
        self, input_ids: torch.Tensor, pad_token_id: int
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        B = input_ids.size(0)
        logger.debug(f"input_ids: {input_ids[0]}")

        # Create attention mask
        attention_mask = (input_ids != pad_token_id).to(dtype=torch.int64)
        logger.debug(f"attention_mask: {attention_mask[0]}")

        attention_mask = torch.cat(
            [
                attention_mask,
                torch.ones(B, self.gist_tokens_mean.size(1), device=input_ids.device),
            ],
            dim=1,
        )

        embeds = self.model.get_input_embeddings()(input_ids)
        embeds = torch.cat(
            [embeds, self.gist_tokens_mean.unsqueeze(0).expand(B, -1, -1)], dim=1
        )

        # Process input through T5 encoder
        encoder_outputs = self.model.encoder(
            inputs_embeds=embeds,
            attention_mask=attention_mask,
        )

        # Get the encoder output
        last_hidden_states = encoder_outputs.last_hidden_state
        mean = last_hidden_states[:, -self.n_gist_tokens :, :]
        logvar = self.gist_tokens_logvar.unsqueeze(0).expand(B, -1, -1)
        # Calculate KL divergence
        kl_loss = self.kl_divergence(mean, logvar) * self.kl_weight
        logger.debug(f"kl_loss: {kl_loss}")

        # Sample latent vectors using reparameterization trick
        gisted_embeds = self.reparameterize(mean, logvar)

        return gisted_embeds, kl_loss, mean

    def push_to_hub(self, repo_id: str, ckpt_dir: str = CKPT_DIR):
        self.model.push_to_hub(repo_id)

        folder = os.path.dirname(f"{ckpt_dir}/{repo_id}/gist_tokens.npy")
        if not os.path.exists(folder):
            os.makedirs(folder)

        # Save tensors using safetensors
        tensors = {
            "gist_tokens_mean": self.gist_tokens_mean.data,
            "gist_tokens_logvar": self.gist_tokens_logvar.data,
        }
        save_path = f"{ckpt_dir}/{repo_id}/latent_tokens.safetensors"
        save_file(tensors, save_path)

        # Save configuration parameters
        config = {
            "n_gist_tokens": self.n_gist_tokens,
            "block_size": self.block_size,
            "kl_weight": self.kl_weight,
        }
        import json

        config_path = f"{ckpt_dir}/{repo_id}/latent_config.json"
        with open(config_path, "w") as f:
            json.dump(config, f)

        # Upload to hub
        hf_api = HfApi()
        hf_api.upload_file(
            repo_id=repo_id,
            path_or_fileobj=save_path,
            path_in_repo="latent_tokens.safetensors",
        )
        # Upload config
        hf_api.upload_file(
            repo_id=repo_id,
            path_or_fileobj=config_path,
            path_in_repo="latent_config.json",
        )

    @classmethod
    def from_pretrained(
        cls,
        repo_id: str,
        torch_dtype: torch.dtype = torch.bfloat16,
        device: str = "cuda",
    ):
        """Create a LatentEncoder instance from a pretrained model on the Hub."""
        # Download config
        try:
            config_path = snapshot_download(
                repo_id=repo_id, allow_patterns=["latent_config.json"]
            )
            config_path = os.path.join(config_path, "latent_config.json")
            import json

            with open(config_path, "r") as f:
                config = json.load(f)
        except Exception as e:
            raise ValueError(f"Could not find latent_config.json in {repo_id}") from e

        # Get model name from repo
        model_name = repo_id

        # Create instance with loaded config
        kl_weight = config.get("kl_weight", 0.1)  # Default if not found in older models

        instance = cls(
            model_name=model_name,
            n_gist_tokens=config["n_gist_tokens"],
            block_size=config["block_size"],
            torch_dtype=torch_dtype,
            kl_weight=kl_weight,
        )

        # Load weights
        instance.load_pretrained(repo_id)
        instance.to(device)
        return instance

    def load_pretrained(self, repo_id: str):
        # Download safetensors file
        tokens_path = snapshot_download(
            repo_id=repo_id,
            allow_patterns=["latent_tokens.safetensors"],
            force_download=True,
        )
        tokens_path = os.path.join(tokens_path, "latent_tokens.safetensors")
        print(f"Loading latent tokens from {tokens_path}")
        # Load tensors using safetensors
        if os.path.exists(tokens_path):
            tensors = load_file(tokens_path)
            if "gist_tokens_mean" in tensors:
                # New VAE format
                self.gist_tokens_mean.data = tensors["gist_tokens_mean"]
                self.gist_tokens_logvar.data = tensors["gist_tokens_logvar"]
            else:
                # Backward compatibility with older non-VAE format
                self.gist_tokens_mean.data = tensors["gist_tokens"]
                # Initialize logvar with zeros
                self.gist_tokens_logvar.data = torch.zeros_like(
                    self.gist_tokens_mean.data
                )

            print(f"Loaded latent tokens from {tokens_path}")
        else:
            raise ValueError(f"Could not find latent_tokens.safetensors in {repo_id}")


class LatentDecoder(nn.Module):
    def __init__(
        self,
        model_name: str,
        n_gist_tokens: int,
        block_size: int,
        torch_dtype: torch.dtype = torch.bfloat16,
    ):
        super().__init__()
        self.model = T5ForConditionalGeneration.from_pretrained(
            model_name, torch_dtype=torch_dtype
        )
        self.base_config = self.model.config
        self.mem_size = n_gist_tokens
        self.block_size = block_size
        self.torch_dtype = torch_dtype

        # Create a projection layer if needed to match embedding dimensions
        if hasattr(self.base_config, "d_model"):
            self.hidden_size = self.base_config.d_model
        else:
            self.hidden_size = self.base_config.hidden_size

        # T5 expects encoder hidden states as cross-attention input
        # This layer will project our latent embeddings to encoder hidden states
        self.latent_to_encoder_proj = nn.Linear(
            self.hidden_size, self.hidden_size, dtype=torch_dtype
        )

    def forward(
        self,
        input_ids: torch.Tensor,
        mem_embeds: torch.Tensor,
        labels: Optional[torch.Tensor] = None,
        ignore_index: int = -100,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[torch.Tensor]]:
        B, T = input_ids.size()
        logger.debug(f"input_ids: {input_ids[0]}")
        logger.debug(f"mem shape: {mem_embeds.shape}")

        # Ensure mem_embeds has the correct dtype
        mem_embeds = mem_embeds.to(dtype=self.torch_dtype)

        # Project memory embeddings to encoder hidden states format
        encoder_hidden_states = self.latent_to_encoder_proj(mem_embeds)

        # Create a proper BaseModelOutput object for encoder_outputs
        from transformers.modeling_outputs import BaseModelOutput

        encoder_outputs = BaseModelOutput(last_hidden_state=encoder_hidden_states)

        # Create attention mask for decoder inputs
        decoder_attention_mask = (input_ids != self.model.config.pad_token_id).long()

        # Run the T5 model with encoder_outputs to use cross-attention
        # between decoder and our projected memory embeddings
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=decoder_attention_mask,
            encoder_outputs=encoder_outputs,
            labels=labels if labels is not None else None,
            return_dict=True,
        )

        # Get logits
        logits = outputs.logits

        if labels is not None:
            # Use the loss directly from the T5 model
            loss = outputs.loss

            # Calculate token accuracy
            predictions = torch.argmax(logits, dim=-1)
            valid_tokens = labels != ignore_index
            correct_tokens = torch.eq(predictions, labels) & valid_tokens
            accuracy = (
                torch.sum(correct_tokens).float() / torch.sum(valid_tokens).float()
            )

            return logits, loss, accuracy

        return logits, None, None

    def generate(
        self,
        mem_embeds: torch.Tensor,
        max_new_tokens: int,
        temperature: float = 0.7,
        **kwargs,
    ) -> torch.Tensor:
        """Generate text using T5's built-in generation method with our memory embeddings."""
        B = mem_embeds.size(0)
        device = mem_embeds.device

        # Ensure mem_embeds has the correct dtype
        mem_embeds = mem_embeds.to(dtype=self.torch_dtype)

        # Project memory embeddings to encoder hidden states format
        encoder_hidden_states = self.latent_to_encoder_proj(mem_embeds)

        # Create a proper BaseModelOutput object for encoder_outputs
        from transformers.modeling_outputs import BaseModelOutput

        encoder_outputs = BaseModelOutput(last_hidden_state=encoder_hidden_states)

        # Create dummy input for the decoder to start generation
        decoder_input_ids = (
            torch.ones((B, 1), device=device, dtype=torch.long)
            * self.model.config.decoder_start_token_id
        )

        # Use T5's built-in generation with our encoder outputs
        output_ids = self.model.generate(
            decoder_input_ids=decoder_input_ids,
            encoder_outputs=encoder_outputs,
            max_new_tokens=max_new_tokens,
            do_sample=True if temperature > 0 else False,
            temperature=temperature,
            **kwargs,
        )

        return output_ids

    def push_to_hub(self, repo_id: str, ckpt_dir: str = CKPT_DIR):
        self.model.push_to_hub(repo_id)

        folder = os.path.dirname(f"{ckpt_dir}/{repo_id}/decoder_proj.safetensors")
        if not os.path.exists(folder):
            os.makedirs(folder)

        # Save projection layer using safetensors
        tensors = {
            "weight": self.latent_to_encoder_proj.weight.data,
            "bias": (
                self.latent_to_encoder_proj.bias.data
                if self.latent_to_encoder_proj.bias is not None
                else None
            ),
        }
        save_path = f"{ckpt_dir}/{repo_id}/decoder_proj.safetensors"
        save_file(tensors, save_path)

        # Save configuration parameters
        config = {
            "n_gist_tokens": self.mem_size,
            "block_size": self.block_size,
        }
        import json

        config_path = f"{ckpt_dir}/{repo_id}/decoder_config.json"
        with open(config_path, "w") as f:
            json.dump(config, f)

        # Upload to hub
        hf_api = HfApi()
        hf_api.upload_file(
            repo_id=repo_id,
            path_or_fileobj=save_path,
            path_in_repo="decoder_proj.safetensors",
        )
        # Upload config
        hf_api.upload_file(
            repo_id=repo_id,
            path_or_fileobj=config_path,
            path_in_repo="decoder_config.json",
        )

    @classmethod
    def from_pretrained(
        cls,
        repo_id: str,
        torch_dtype: torch.dtype = torch.bfloat16,
        device: str = "cuda",
    ):
        """Create a LatentDecoder instance from a pretrained model on the Hub."""
        # Download config
        try:
            config_path = snapshot_download(
                repo_id=repo_id, allow_patterns=["decoder_config.json"]
            )
            config_path = os.path.join(config_path, "decoder_config.json")
            import json

            with open(config_path, "r") as f:
                config = json.load(f)
        except Exception as e:
            raise ValueError(f"Could not find decoder_config.json in {repo_id}") from e

        # Get model name from repo
        model_name = repo_id

        # Create instance with loaded config
        instance = cls(
            model_name=model_name,
            n_gist_tokens=config["n_gist_tokens"],
            block_size=config["block_size"],
            torch_dtype=torch_dtype,
        )

        # Load projection layer weights
        instance.load_pretrained(repo_id)
        instance.to(device)
        return instance

    def load_pretrained(self, repo_id: str):
        # Download safetensors file
        proj_path = snapshot_download(
            repo_id=repo_id,
            allow_patterns=["decoder_proj.safetensors"],
            force_download=True,
        )
        proj_path = os.path.join(proj_path, "decoder_proj.safetensors")
        print(f"Loading decoder projection from {proj_path}")

        # Load tensors using safetensors
        if os.path.exists(proj_path):
            tensors = load_file(proj_path)
            self.latent_to_encoder_proj.weight.data = tensors["weight"]
            if "bias" in tensors and tensors["bias"] is not None:
                self.latent_to_encoder_proj.bias.data = tensors["bias"]
            print(f"Loaded decoder projection from {proj_path}")
        else:
            raise ValueError(f"Could not find decoder_proj.safetensors in {repo_id}")


class T5LatentVAEPipeline:
    def __init__(
        self,
        pretrained_encoder_id: Optional[str] = None,
        pretrained_decoder_id: Optional[str] = None,
        torch_dtype: torch.dtype = torch.bfloat16,
        device: str = "cuda",
    ):
        """
        Initialize the T5LatentVAEPipeline with pretrained encoder and/or decoder.

        Args:
            pretrained_encoder_id: HuggingFace repo ID for the encoder
            pretrained_decoder_id: HuggingFace repo ID for the decoder
            torch_dtype: Torch dtype for the models
            device: Device to load the models on
        """
        self.pretrained_encoder_id = pretrained_encoder_id
        self.pretrained_decoder_id = pretrained_decoder_id
        self.torch_dtype = torch_dtype
        self.device = device

        self.encoder = None
        self.decoder = None
        self.tokenizer = None

        # Load tokenizer from encoder or decoder model
        if pretrained_encoder_id:
            self.tokenizer = AutoTokenizer.from_pretrained(pretrained_encoder_id)
        elif pretrained_decoder_id:
            self.tokenizer = AutoTokenizer.from_pretrained(pretrained_decoder_id)

    def _load_encoder(self):
        """Load the encoder model if not already loaded."""
        if self.encoder is None and self.pretrained_encoder_id:
            self.encoder = LatentEncoder.from_pretrained(
                self.pretrained_encoder_id,
                torch_dtype=self.torch_dtype,
                device=self.device,
            )
            self.encoder.eval()

    def _load_decoder(self):
        """Load the decoder model if not already loaded."""
        if self.decoder is None and self.pretrained_decoder_id:
            # Get model name from repo ID
            model_name = self.pretrained_decoder_id

            # Download config to get parameters
            config_path = snapshot_download(
                repo_id=self.pretrained_encoder_id,
                allow_patterns=["latent_config.json"],
            )
            config_path = os.path.join(config_path, "latent_config.json")
            import json

            with open(config_path, "r") as f:
                config = json.load(f)

            self.decoder = LatentDecoder(
                model_name=model_name,
                n_gist_tokens=config["n_gist_tokens"],
                block_size=config["block_size"],
                torch_dtype=self.torch_dtype,
            )
            self.decoder.eval()
            self.decoder.to(self.device)

    def encode(self, text: str) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Encode text to latent embeddings.

        Args:
            text: Input text to encode

        Returns:
            Memory embeddings tensor and mean tensor
        """
        if not self.pretrained_encoder_id:
            raise ValueError("Encoder model ID not provided during initialization")

        self._load_encoder()

        # Tokenize input
        inputs = self.tokenizer(
            text,
            return_tensors="pt",
            padding="max_length",
            max_length=self.encoder.block_size,
            truncation=True,
        )
        input_ids = inputs.input_ids.to(self.encoder.model.device)
        print(input_ids)

        # Generate memory embeddings
        with torch.no_grad():
            mem_embeds, _, mean = self.encoder(input_ids, self.tokenizer.pad_token_id)

        return mem_embeds, mean

    def decode(
        self,
        mem_embeds: torch.Tensor,
        max_new_tokens: int = 100,
        temperature: float = 0.7,
        **kwargs,
    ) -> str:
        """
        Decode latent embeddings to text.

        Args:
            mem_embeds: Memory embeddings from encoder
            max_new_tokens: Maximum number of tokens to generate
            temperature: Temperature for sampling

        Returns:
            Generated text
        """
        if not self.pretrained_decoder_id:
            raise ValueError("Decoder model ID not provided during initialization")

        self._load_decoder()

        # Move embeddings to the same device as the decoder
        mem_embeds = mem_embeds.to(self.decoder.model.device)

        # Generate tokens
        with torch.no_grad():
            output_ids = self.decoder.generate(
                mem_embeds,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                **kwargs,
            )

        # Decode output tokens to text
        output_text = self.tokenizer.decode(output_ids[0], skip_special_tokens=True)

        return output_text


# Backwards compatibility alias
GPTLatentVAEPipeline = T5LatentVAEPipeline


if __name__ == "__main__":
    # Use a T5 model for demonstration
    model_name = "t5-small"
    n_gist_tokens = 64
    block_size = 512

    # Initialize tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    # Add padding token if not present
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Create encoder and decoder
    latent_encoder = LatentEncoder(model_name, n_gist_tokens, block_size)
    latent_decoder = LatentDecoder(model_name, n_gist_tokens, block_size)

    # Prepare sample input
    sample_text = "This is a sample input to test our T5 latent VAE model."
    input_ids = tokenizer(
        sample_text,
        return_tensors="pt",
        padding="max_length",
        max_length=block_size,
        truncation=True,
    ).input_ids

    # Encode to latent space
    mem_embeds, mean = latent_encoder(input_ids, tokenizer.pad_token_id)
    print(f"Memory embeddings shape: {mem_embeds.shape}")

    # Prepare decoder input (for demonstration)
    decoder_input = tokenizer("", return_tensors="pt").input_ids

    # Decode from latent space
    output_ids = latent_decoder.generate(mem_embeds, max_new_tokens=50)
    generated_text = tokenizer.decode(output_ids[0], skip_special_tokens=True)

    print("Generated text:", generated_text)
