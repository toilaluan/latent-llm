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
        n_gist_tokens: int,
        n_ae_tokens: int,
        block_size: int,
        torch_dtype: torch.dtype = torch.bfloat16,
        use_lora: bool = False,
        lora_r: int = 8,
        lora_alpha: int = 16,
        lora_dropout: float = 0.05,
        lora_target_modules: Optional[list] = None,
    ):
        super().__init__()
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch_dtype,
        )
        self.base_config = self.model.config

        # Apply LoRA if specified
        self.use_lora = use_lora
        if use_lora:
            if lora_target_modules is None:
                # Default target modules for transformer models
                lora_target_modules = ["q_proj", "v_proj"]

            lora_config = LoraConfig(
                task_type=TaskType.CAUSAL_LM,
                inference_mode=False,
                r=lora_r,
                lora_alpha=lora_alpha,
                lora_dropout=lora_dropout,
                target_modules=lora_target_modules,
            )
            self.model = get_peft_model(self.model, lora_config)
            self.model.print_trainable_parameters()

        self.gist_tokens = nn.Parameter(
            torch.randn(n_gist_tokens, self.base_config.hidden_size, dtype=torch_dtype)
        )
        self.n_gist_tokens = n_gist_tokens
        self.ae_tokens = nn.Parameter(
            torch.randn(n_ae_tokens, self.base_config.hidden_size, dtype=torch_dtype)
        )
        self.block_size = block_size
        self.init_weights()
        self.init_position_ids()

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
        self.register_buffer("position_ids", position_ids)

    def init_weights(self):
        torch.nn.init.kaiming_normal_(self.gist_tokens)

    def get_trainable_parameters(self):
        if self.use_lora:
            return [self.gist_tokens, self.ae_tokens] + [
                p for p in self.model.parameters() if p.requires_grad
            ]
        else:
            return [self.gist_tokens, self.ae_tokens, *self.model.parameters()]

    def forward(
        self, input_ids: torch.Tensor, pad_token_id: int
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        B = input_ids.size(0)
        embeds = self.model.get_input_embeddings()(input_ids)
        position_ids = self.position_ids.repeat(B, 1)
        masks = input_ids != pad_token_id
        embeds = torch.cat(
            [
                embeds,
                self.gist_tokens.repeat(B, 1, 1),
            ],
            dim=1,
        )
        masks = torch.cat(
            [masks, torch.ones(B, self.n_gist_tokens, device=masks.device)], dim=1
        )
        last_hidden_states = self.model(
            inputs_embeds=embeds,
            output_hidden_states=True,
            attention_mask=masks,
            position_ids=position_ids,
        ).hidden_states[-1]
        gisted_embeds = last_hidden_states[:, -self.n_gist_tokens :, :]
        ae_embeds = self.ae_tokens.repeat(B, 1, 1)
        return torch.cat([gisted_embeds, ae_embeds], dim=1)

    def push_to_hub(self, repo_id: str, ckpt_dir: str = CKPT_DIR):
        if self.use_lora:
            self.model.save_pretrained(f"{ckpt_dir}/{repo_id}")
            # Upload the PEFT adapter to Hub
            self.model.push_to_hub(repo_id)
        else:
            self.model.push_to_hub(repo_id)

        folder = os.path.dirname(f"{ckpt_dir}/{repo_id}/gist_tokens.npy")
        if not os.path.exists(folder):
            os.makedirs(folder)

        # Save tensors using safetensors
        tensors = {
            "gist_tokens": self.gist_tokens.data,
            "ae_tokens": self.ae_tokens.data,
        }
        save_path = f"{ckpt_dir}/{repo_id}/latent_tokens.safetensors"
        save_file(tensors, save_path)

        # Save configuration parameters
        config = {
            "n_gist_tokens": self.n_gist_tokens,
            "n_ae_tokens": self.ae_tokens.size(0),
            "block_size": self.block_size,
            "use_lora": self.use_lora,
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
    def from_pretrained(cls, repo_id: str, torch_dtype: torch.dtype = torch.bfloat16):
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
        instance = cls(
            model_name=model_name,
            n_gist_tokens=config["n_gist_tokens"],
            n_ae_tokens=config["n_ae_tokens"],
            block_size=config["block_size"],
            torch_dtype=torch_dtype,
            use_lora=False,  # Default to False if not specified
        )

        # Load weights
        instance.load_pretrained(repo_id)
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
            self.gist_tokens.data = tensors["gist_tokens"]
            self.ae_tokens.data = tensors["ae_tokens"]
            print(f"Loaded latent tokens from {tokens_path}")
        else:
            raise ValueError(f"Could not find latent_tokens.safetensors in {repo_id}")


class LatentDecoder(nn.Module):
    def __init__(
        self,
        model_name: str,
        n_gist_tokens: int,
        n_ae_tokens: int,
        block_size: int,
        torch_dtype: torch.dtype = torch.bfloat16,
    ):
        super().__init__()
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name, torch_dtype=torch_dtype
        )
        self.base_config = self.model.config
        self.mem_size = n_gist_tokens + n_ae_tokens
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
        embeds = self.model.get_input_embeddings()(input_ids)
        embeds = torch.cat(
            [
                mem_embeds,
                embeds,
            ],
            dim=1,
        )
        position_ids = self.position_ids[: embeds.size(1)].repeat(B, 1)
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
        device = self.model.device
        generated_ids = torch.zeros((B, 0), dtype=torch.long, device=device)
        # Create attention mask (1 for all tokens)
        attention_mask = torch.ones(
            (B, embeds.size(1)), dtype=torch.long, device=device
        )
        position_ids = self.position_ids[: embeds.size(1)].repeat(B, 1)
        max_new_tokens = min(max_new_tokens, self.block_size - embeds.size(1))
        # Generate tokens one by one
        for _ in range(max_new_tokens):
            # Forward pass
            outputs = self.model(
                inputs_embeds=embeds,
                attention_mask=attention_mask,
                # position_ids=position_ids,
            )
            logits = outputs.logits[:, -1, :] / max(temperature, 1e-7)

            # Sample from the distribution
            if temperature > 0:
                probs = torch.softmax(logits, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1).squeeze(-1)
            else:
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

            # Update attention mask
            attention_mask = torch.cat(
                [attention_mask, torch.ones((B, 1), dtype=torch.long, device=device)],
                dim=1,
            )
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
                self.pretrained_encoder_id, torch_dtype=self.torch_dtype
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
                n_ae_tokens=config["n_ae_tokens"],
                block_size=config["block_size"],
                torch_dtype=self.torch_dtype,
            )
            self.decoder.eval()

    def encode(self, text: str) -> torch.Tensor:
        """
        Encode text to latent embeddings.

        Args:
            text: Input text to encode

        Returns:
            Memory embeddings tensor
        """
        if not self.pretrained_encoder_id:
            raise ValueError("Encoder model ID not provided during initialization")

        self._load_encoder()

        # Tokenize input
        inputs = self.tokenizer(text, return_tensors="pt", padding=True)
        input_ids = inputs.input_ids.to(self.encoder.model.device)

        # Generate memory embeddings
        with torch.no_grad():
            mem_embeds = self.encoder(input_ids, self.tokenizer.pad_token_id)

        return mem_embeds

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
    n_ae_tokens = 1
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    latent_encoder = LatentEncoder(model_name, n_gist_tokens, n_ae_tokens)
    latent_decoder = LatentDecoder(model_name, n_gist_tokens, n_ae_tokens)

    input_ids = torch.randint(0, 100, (1, 10))
    mem_embeds = latent_encoder(input_ids)
    logits, loss, accuracy = latent_decoder(input_ids, mem_embeds, labels=input_ids)
    print(loss)
    print(accuracy)

    completion = latent_decoder.generate(mem_embeds, input_ids, max_new_tokens=10)
    print(tokenizer.decode(completion[0]))
