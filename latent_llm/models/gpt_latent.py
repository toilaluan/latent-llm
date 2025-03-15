from transformers import AutoModelForCausalLM, AutoTokenizer
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
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch_dtype,
            attn_implementation="flash_attention_2",
        )
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.base_config = self.model.config

        # For VAE, we need separate parameters for mean and log variance
        self.gist_tokens_mean = nn.Parameter(
            torch.randn(n_gist_tokens, self.base_config.hidden_size, dtype=torch_dtype)
        )
        self.gist_tokens_logvar = nn.Parameter(
            torch.zeros(n_gist_tokens, self.base_config.hidden_size, dtype=torch_dtype)
        )
        self.n_gist_tokens = n_gist_tokens
        self.block_size = block_size
        self.latent_config = {
            "n_gist_tokens": n_gist_tokens,
            "block_size": block_size,
        }
        self.kl_weight = kl_weight
        self.init_weights()
        self.init_position_ids()
        self.enable_debug_logging = enable_debug_logging
        if enable_debug_logging:
            _enable_debug_logging()

    def init_position_ids(self):
        mem_pos_step = max(self.block_size // self.n_gist_tokens, 1)
        memory_position_ids = torch.arange(self.n_gist_tokens) * mem_pos_step
        position_ids = torch.cat(
            [
                torch.arange(self.block_size),
                memory_position_ids,
            ],
            dim=0,
        )
        gist_masks = torch.ones(self.n_gist_tokens, dtype=torch.int64)
        self.register_buffer("position_ids", position_ids)
        self.register_buffer("gist_masks", gist_masks)

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
        Output shape: scalar (averaged over batch and tokens, summed over hidden dimensions)
        """
        # Calculate KL divergence per token
        # Shape: (batch_size, n_gist_tokens, hidden_size)
        kl_per_element = 0.5 * (mean.pow(2) + logvar.exp() - logvar - 1)

        # Sum over hidden dimensions, then average over tokens and batch
        # First sum over hidden dimension
        kl_per_token = torch.sum(kl_per_element, dim=-1)  # (batch_size, n_gist_tokens)

        # Then average over tokens and batch
        return torch.mean(kl_per_token)

    def forward(
        self, input_ids: torch.Tensor, pad_token_id: int
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        B = input_ids.size(0)
        logger.debug(f"input_ids: {input_ids[0]}")
        embeds = self.model.get_input_embeddings()(input_ids)
        masks = (input_ids != pad_token_id).to(dtype=torch.int64)
        logger.debug(f"masks: {masks[0]}")
        # Use gist_tokens_mean as initial tokens for model processing
        gist_tokens = self.gist_tokens_mean.unsqueeze(0).expand(B, -1, -1)

        embeds = torch.cat(
            [
                embeds,
                gist_tokens,
            ],
            dim=1,
        )
        position_ids = self.position_ids[: embeds.size(1)].repeat(B, 1)
        masks = torch.cat([masks, self.gist_masks.repeat(B, 1)], dim=1)
        logger.debug(f"position_ids: {position_ids[0]}")
        logger.debug(f"concatenated masks: {masks[0]}")

        last_hidden_states = self.model(
            inputs_embeds=embeds,
            output_hidden_states=True,
            attention_mask=masks,
            position_ids=position_ids,
        ).hidden_states[-1]

        # Get the hidden states for gist tokens
        gisted_hidden = last_hidden_states[:, -self.n_gist_tokens :, :]

        # For VAE, we interpret these as parameters of the distribution
        mean = gisted_hidden
        logvar = self.gist_tokens_logvar.unsqueeze(0).expand(B, -1, -1)
        logger.debug(f"mean: {mean[0]}")
        logger.debug(f"logvar: {logvar[0]}")

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
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name, torch_dtype=torch_dtype, attn_implementation="flash_attention_2"
        )
        self.base_config = self.model.config
        self.mem_size = n_gist_tokens
        self.block_size = block_size
        self.init_position_ids()

    def init_position_ids(self):
        mem_pos_step = max(self.block_size // self.mem_size, 1)
        memory_position_ids = torch.arange(self.mem_size) * mem_pos_step
        position_ids = torch.cat(
            [
                memory_position_ids,
                torch.arange(self.block_size),
            ],
            dim=0,
        )
        self.register_buffer("position_ids", position_ids)

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
        embeds = self.model.get_input_embeddings()(input_ids)
        embeds = torch.cat(
            [
                mem_embeds,
                embeds,
            ],
            dim=1,
        )
        logger.debug(f"embeds shape: {embeds[0]}")
        position_ids = self.position_ids[: embeds.size(1)].repeat(B, 1)
        logger.debug(f"position_ids shape: {position_ids[0]}")
        logits = self.model(
            inputs_embeds=embeds,
            # position_ids=position_ids,
        ).logits
        # labels = [a b c d], mem_embeds = [m m m m]
        # input_ids: [m m m m a b c d] -> predicts [x x x x a b c d]
        # label map: [x a b c] -> [a b c d]
        logits = logits[:, mem_embeds.size(1) - 1 : -1, :]
        if labels is not None:
            loss = F.cross_entropy(
                logits.reshape(-1, logits.size(-1)),
                labels.reshape(-1),
                ignore_index=ignore_index,
            )

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
        embeds: torch.Tensor,
        max_new_tokens: int,
        temperature: float = 0.01,
        **kwargs,
    ) -> torch.Tensor:
        """Generate text using a more efficient approach with temperature sampling."""
        embeds = embeds.to(self.model.dtype)
        B = embeds.size(0)
        device = embeds.device
        generated_ids = torch.zeros((B, 0), dtype=torch.long, device=device)
        # Create attention mask (1 for all tokens)
        attention_mask = torch.ones(
            (B, embeds.size(1)), dtype=torch.long, device=device
        )
        position_ids = self.position_ids[: embeds.size(1)].repeat(B, 1)
        max_new_tokens = min(max_new_tokens, self.position_ids.size(0) - embeds.size(1))
        print("max_new_tokens", max_new_tokens)

        # Generate tokens one by one
        for _ in range(max_new_tokens):
            # Forward pass
            outputs = self.model(
                inputs_embeds=embeds,
                attention_mask=attention_mask,
                # position_ids=position_ids,
            )
            logits = outputs.logits[:, -1, :]
            # Apply temperature scaling
            scaled_logits = logits / max(temperature, 1e-7)

            # Sample from the distribution
            if temperature > 0:
                probs = torch.softmax(scaled_logits, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1)
            else:
                next_token = torch.argmax(scaled_logits, dim=-1, keepdim=True)

            # Check if all sequences have EOS
            if (next_token == self.model.config.eos_token_id).all():
                break

            # Append new tokens
            if generated_ids.size(1) == 0:
                generated_ids = next_token
            else:
                generated_ids = torch.cat([generated_ids, next_token], dim=1)

            # Update input embeddings for next iteration
            next_token_embeds = self.model.get_input_embeddings()(next_token)
            embeds = torch.cat([embeds, next_token_embeds], dim=1)

            # Update attention mask
            new_mask = torch.ones((B, 1), dtype=torch.long, device=device)
            attention_mask = torch.cat([attention_mask, new_mask], dim=1)
            position_ids = self.position_ids[: embeds.size(1)].repeat(B, 1)

        return generated_ids

    def push_to_hub(self, repo_id: str):
        self.model.push_to_hub(repo_id)


class GPTLatentVAEPipeline:
    def __init__(
        self,
        pretrained_encoder_id: Optional[str] = None,
        pretrained_decoder_id: Optional[str] = None,
        torch_dtype: torch.dtype = torch.bfloat16,
        device: str = "cuda",
    ):
        """
        Initialize the GPTLatentPipeline with pretrained encoder and/or decoder.

        Args:
            pretrained_encoder_id: HuggingFace repo ID for the encoder
            pretrained_decoder_id: HuggingFace repo ID for the decoder
            torch_dtype: Torch dtype for the models
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
            # Get model name from repo ID (assuming format is consistent with LatentEncoder.from_pretrained)
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


if __name__ == "__main__":
    model_name = "HuggingFaceTB/SmolLM2-135M"
    n_gist_tokens = 64
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    latent_encoder = LatentEncoder(model_name, n_gist_tokens)
    latent_decoder = LatentDecoder(model_name, n_gist_tokens)

    input_ids = torch.randint(0, 100, (1, 10))
    mem_embeds, mean = latent_encoder(input_ids)
    logits, loss, _ = latent_decoder(input_ids, mem_embeds, labels=input_ids)
    print(loss)

    completion = latent_decoder.generate(mem_embeds, input_ids, max_new_tokens=10)
    print(tokenizer.decode(completion[0]))
