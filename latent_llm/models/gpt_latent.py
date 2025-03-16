from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple
import os
from huggingface_hub import HfApi
import numpy as np
from huggingface_hub import snapshot_download
from safetensors.torch import save_file, load_file
from peft import get_peft_model, LoraConfig, TaskType, PeftModel


CKPT_DIR = ".training_cache/checkpoints"

if not os.path.exists(CKPT_DIR):
    os.makedirs(CKPT_DIR)


class LatentEncoder(nn.Module):
    def __init__(
        self,
        model_name: str,
        latent_size: int,
        block_size: int,
        torch_dtype: torch.dtype = torch.bfloat16,
        kl_weight: float = 1e-4,
        attn_implementation: str = "flash_attention_2",
    ):
        super().__init__()
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch_dtype,
            attn_implementation=attn_implementation,
        )
        self.base_config = self.model.config
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        # For VAE, we need separate parameters for mean and log variance
        self.latent_tokens_mean = nn.Parameter(
            torch.randn(latent_size, self.base_config.hidden_size, dtype=torch_dtype)
        )
        self.latent_tokens_logvar = nn.Parameter(
            torch.zeros(latent_size, self.base_config.hidden_size, dtype=torch_dtype)
        )
        self.latent_size = latent_size
        self.block_size = block_size
        self.latent_config = {
            "latent_size": latent_size,
            "block_size": block_size,
        }
        self.kl_weight = kl_weight
        self.init_weights()
        self.init_position_ids()
        self.init_lora()

    def init_lora(self):
        lora_config = LoraConfig(
            r=128,
            lora_alpha=256,
            lora_dropout=0.1,
            target_modules=["k_proj", "v_proj", "q_proj"],
            task_type=TaskType.CAUSAL_LM,
        )
        self.model = get_peft_model(self.model, lora_config)
        self.model.print_trainable_parameters()

    def init_position_ids(self):
        step = max(self.block_size // self.latent_size, 1)
        position_ids = torch.arange(self.latent_size) * step
        position_ids = torch.cat(
            [
                position_ids,
                torch.arange(self.block_size),
            ],
            dim=0,
        )
        self.register_buffer("position_ids", position_ids)

    def init_weights(self):
        torch.nn.init.kaiming_normal_(self.latent_tokens_mean)
        torch.nn.init.zeros_(self.latent_tokens_logvar)

    def get_trainable_parameters(self):
        return [
            self.latent_tokens_mean,
            self.latent_tokens_logvar,
            *self.model.parameters(),
        ]

    def reparameterize(self, mean, logvar):
        """
        Reparameterization trick: sample from a standard normal and scale by
        standard deviation and shift by mean for backpropagation.
        """
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mean + eps * std

    def kl_divergence(self, mean, logvar):
        # Sum across hidden dimension
        kl_per_latent = torch.sum(1 + logvar - mean.pow(2) - logvar.exp(), dim=2)
        # Mean across latent dimension first
        kl_per_batch = torch.mean(kl_per_latent, dim=1)
        # Then mean across batch dimension
        return -0.5 * torch.mean(kl_per_batch, dim=0)

    def forward(
        self, input_ids: torch.Tensor, pad_token_id: int
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        B = input_ids.size(0)
        context_embeds = self.model.get_input_embeddings()(input_ids)
        masks = input_ids != pad_token_id

        # Use expand instead of repeat for more memory efficiency
        latent_tokens = self.latent_tokens_mean.unsqueeze(0).expand(B, -1, -1)

        inputs_embeds = torch.cat(
            [
                context_embeds,
                latent_tokens,
            ],
            dim=1,
        )
        masks = torch.cat(
            [masks, torch.ones(B, self.latent_size, device=masks.device)], dim=1
        )
        position_ids = (
            self.position_ids[: inputs_embeds.size(1)].unsqueeze(0).expand(B, -1)
        )
        last_hidden_states = self.model(
            inputs_embeds=inputs_embeds,
            output_hidden_states=True,
            attention_mask=masks,
            position_ids=position_ids,
        ).hidden_states[-1]

        # Get the hidden states for gist tokens
        latents = last_hidden_states[:, -self.latent_size :, :]

        # latents = self.latent_transformer(latents)

        logvar = self.latent_tokens_logvar.unsqueeze(0).expand(B, -1, -1)

        if self.kl_weight > 0:
            # Calculate KL divergence
            kl_loss = self.kl_divergence(latents, logvar) * self.kl_weight
            rep_latent_embeds = self.reparameterize(latents, logvar)
        else:
            kl_loss = torch.zeros(1, device=latents.device)
            rep_latent_embeds = latents

        return rep_latent_embeds, kl_loss, latents

    def push_to_hub(self, repo_id: str, ckpt_dir: str = CKPT_DIR):
        self.model.push_to_hub(repo_id)

        # Create full directory path for the checkpoint
        full_ckpt_path = os.path.join(ckpt_dir, repo_id)
        os.makedirs(full_ckpt_path, exist_ok=True)

        # Save tensors using safetensors
        tensors = {
            "latent_tokens_mean": self.latent_tokens_mean.data,
            "latent_tokens_logvar": self.latent_tokens_logvar.data,
        }
        save_path = os.path.join(full_ckpt_path, "latent_tokens.safetensors")
        save_file(tensors, save_path)

        # Save configuration parameters
        config = {
            "latent_size": self.latent_size,
            "block_size": self.block_size,
            "kl_weight": self.kl_weight,
        }
        import json

        config_path = os.path.join(full_ckpt_path, "latent_config.json")
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
            latent_size=config["latent_size"],
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
            if "latent_tokens_mean" in tensors:
                # New VAE format
                self.latent_tokens_mean.data = tensors["latent_tokens_mean"]
                self.latent_tokens_logvar.data = tensors["latent_tokens_logvar"]
            else:
                # Backward compatibility with older non-VAE format
                self.latent_tokens_mean.data = tensors["latent_tokens"]
                # Initialize logvar with zeros
                self.latent_tokens_logvar.data = torch.zeros_like(
                    self.latent_tokens_mean.data
                )

            print(f"Loaded latent tokens from {tokens_path}")
        else:
            raise ValueError(f"Could not find latent_tokens.safetensors in {repo_id}")


class LatentProjector(nn.Module):
    def __init__(
        self,
        hidden_size: int,
        latent_size: int,
        torch_dtype: torch.dtype = torch.bfloat16,
    ):
        # mlp style with residual connection
        super().__init__()
        self.linear1 = nn.Linear(
            hidden_size, latent_size, dtype=torch_dtype, bias=False
        )
        self.linear2 = nn.Linear(
            latent_size, latent_size, dtype=torch_dtype, bias=False
        )
        self.ln_1 = nn.LayerNorm(latent_size, dtype=torch_dtype)
        self.ln_2 = nn.LayerNorm(latent_size, dtype=torch_dtype)

    def _init_weights(self):
        # std 0.02
        nn.init.normal_(self.linear1.weight, std=0.02)
        nn.init.normal_(self.linear2.weight, std=0.02)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear2(x + self.ln_2(self.linear1(self.ln_1(x))))


class LatentDecoder(nn.Module):
    def __init__(
        self,
        model_name: str,
        latent_size: int,
        block_size: int,
        torch_dtype: torch.dtype = torch.bfloat16,
        attn_implementation: str = "flash_attention_2",
        mid_token_size: int = 16,
    ):
        super().__init__()
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch_dtype,
            attn_implementation=attn_implementation,
        )
        self.base_config = self.model.config
        self.latent_size = latent_size
        self.block_size = block_size
        # self.mid_tokens = nn.Parameter(
        #     torch.randn(mid_token_size, self.base_config.hidden_size, dtype=torch_dtype)
        # )
        # self.mid_token_size = mid_token_size
        # self.latent_proj = LatentProjector(
        #     self.base_config.hidden_size, self.base_config.hidden_size, torch_dtype
        # )
        self.init_position_ids()

    def freeze_base_model(self):
        for param in self.model.parameters():
            param.requires_grad = False

    def init_position_ids(self):
        step = max(self.block_size // self.latent_size, 1)
        position_ids = torch.arange(self.latent_size) * step
        position_ids = torch.cat(
            [
                position_ids,
                torch.arange(self.block_size),
            ],
            dim=0,
        )
        self.register_buffer("position_ids", position_ids)

    def forward(
        self,
        input_ids: torch.Tensor,
        latent_embeds: torch.Tensor,
        labels: Optional[torch.Tensor] = None,
        ignore_index: int = -100,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[torch.Tensor]]:
        B, T = input_ids.size()
        # latent_embeds = self.latent_proj(latent_embeds)
        context_embeds = self.model.get_input_embeddings()(input_ids)
        embeds = torch.cat(
            [
                latent_embeds,
                # self.mid_tokens.unsqueeze(0).expand(B, -1, -1),
                context_embeds,
            ],
            dim=1,
        )
        logits = self.model(
            inputs_embeds=embeds,
            # position_ids=self.position_ids[: embeds.size(1)].unsqueeze(0).expand(B, -1),
        ).logits
        logits = logits[:, self.latent_size - 1 : -1, :]
        if labels is not None:
            loss = F.cross_entropy(
                logits.reshape(-1, logits.size(-1)),
                labels.reshape(-1),
                ignore_index=ignore_index,
                reduction="mean",
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
        latent_embeds: torch.Tensor,
        max_new_tokens: int,
        **kwargs,
    ) -> torch.Tensor:
        """Generate text using a more efficient approach with temperature sampling."""
        latent_embeds = latent_embeds.to(self.model.dtype)
        B = latent_embeds.size(0)
        device = self.model.device
        generated_ids = torch.zeros((B, 0), dtype=torch.long, device=device)
        # Create attention mask (1 for all tokens)
        max_new_tokens = min(
            max_new_tokens, self.position_ids.size(0) - latent_embeds.size(1)
        )
        embeds = torch.cat(
            [
                latent_embeds,
                # self.mid_tokens.unsqueeze(0).expand(B, -1, -1),
            ],
            dim=1,
        )
        # Generate tokens one by one
        for _ in range(max_new_tokens):
            # Forward pass
            outputs = self.model(
                inputs_embeds=embeds,
            )
            logits = outputs.logits[:, -1, :]
            next_token = torch.argmax(logits, dim=-1)

            # Stop if all sequences have EOS
            if (next_token == self.model.config.eos_token_id).all():
                break

            # Append new tokens - handle both empty and non-empty cases
            next_token = next_token.unsqueeze(-1)
            if generated_ids.size(1) == 0:
                generated_ids = next_token
            else:
                generated_ids = torch.cat([generated_ids, next_token], dim=-1)

            # Update input embeddings for next iteration
            next_token_embeds = self.model.get_input_embeddings()(next_token)
            embeds = torch.cat([embeds, next_token_embeds], dim=1)
        return generated_ids

    def push_to_hub(self, repo_id: str):
        self.model.push_to_hub(repo_id)

        # # Create full directory path for the checkpoint
        # full_ckpt_path = os.path.join(CKPT_DIR, repo_id)
        # os.makedirs(full_ckpt_path, exist_ok=True)

        # # Save decoder-specific tensors using safetensors
        # tensors = {
        #     **self._get_proj_state_dict(),
        # }
        # save_path = os.path.join(full_ckpt_path, "decoder_tokens.safetensors")
        # save_file(tensors, save_path)

        # # Save decoder configuration
        # config = {
        #     "latent_size": self.latent_size,
        #     "block_size": self.block_size,
        #     "mid_token_size": self.mid_token_size,
        # }
        # import json

        # config_path = os.path.join(full_ckpt_path, "decoder_config.json")
        # with open(config_path, "w") as f:
        #     json.dump(config, f)

        # # Upload to hub
        # hf_api = HfApi()
        # hf_api.upload_file(
        #     repo_id=repo_id,
        #     path_or_fileobj=save_path,
        #     path_in_repo="decoder_tokens.safetensors",
        # )
        # hf_api.upload_file(
        #     repo_id=repo_id,
        #     path_or_fileobj=config_path,
        #     path_in_repo="decoder_config.json",
        # )

    def _get_proj_state_dict(self):
        """Extract projection layer parameters with clear naming"""
        return {
            "proj.0.weight": self.latent_proj[0].weight.data,
            "proj.0.bias": self.latent_proj[0].bias.data,
            "proj.2.weight": self.latent_proj[2].weight.data,
            "proj.2.bias": self.latent_proj[2].bias.data,
        }

    def load_pretrained(self, repo_id: str):
        # Download safetensors file
        tokens_path = snapshot_download(
            repo_id=repo_id,
            allow_patterns=["decoder_tokens.safetensors"],
            force_download=True,
        )
        tokens_path = os.path.join(tokens_path, "decoder_tokens.safetensors")

        if os.path.exists(tokens_path):
            tensors = load_file(tokens_path)
            self.mid_tokens.data = tensors["mid_tokens"]

            # Reconstruct state dict from named parameters
            proj_state_dict = {
                k.replace("proj.", ""): v
                for k, v in tensors.items()
                if k.startswith("proj.")
            }
            self.latent_proj.load_state_dict(proj_state_dict)
            print(f"Loaded decoder weights from {tokens_path}")
        else:
            raise ValueError(f"Could not find decoder_tokens.safetensors in {repo_id}")


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

        # Generate memory embeddings
        with torch.no_grad():
            rep_latent_embeds, _, latent_embeds = self.encoder(
                input_ids, self.tokenizer.pad_token_id
            )

        return rep_latent_embeds, latent_embeds

    def decode(
        self,
        latent_embeds: torch.Tensor,
        max_new_tokens: int = 100,
    ) -> str:
        if not self.pretrained_decoder_id:
            raise ValueError("Decoder model ID not provided during initialization")

        self._load_decoder()

        # Move embeddings to the same device as the decoder
        latent_embeds = latent_embeds.to(self.decoder.model.device)

        # Generate tokens
        with torch.no_grad():
            output_ids = self.decoder.generate(
                latent_embeds,
                max_new_tokens=max_new_tokens,
            )

        # Decode output tokens to text
        output_text = self.tokenizer.decode(output_ids[0], skip_special_tokens=True)

        return output_text


if __name__ == "__main__":
    model_name = "HuggingFaceTB/SmolLM2-135M"
    latent_size = 16
    block_size = 32
    if torch.cuda.is_available():
        device = "cuda"
        attn_implementation = "flash_attention_2"
    else:
        device = "cpu"
        attn_implementation = "sdpa"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.add_special_tokens({"pad_token": "<|pad|>"})
    latent_encoder = LatentEncoder(
        model_name,
        latent_size,
        block_size,
        attn_implementation=attn_implementation,
    )
    latent_decoder = LatentDecoder(
        model_name,
        latent_size,
        block_size,
        attn_implementation=attn_implementation,
    )
    input_ids = torch.randint(0, 100, (1, 10))
    rep_latent_embeds, kl_loss, latent_embeds = latent_encoder(
        input_ids, tokenizer.pad_token_id
    )
    print(kl_loss)
    logits, loss, _ = latent_decoder(input_ids, latent_embeds, labels=input_ids)
    print(loss)

    completion = latent_decoder.generate(latent_embeds, max_new_tokens=10)
    print(tokenizer.decode(completion[0]))
