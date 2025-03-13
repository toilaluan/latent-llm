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
            use_lora=config.get("use_lora", False),  # Default to False if not specified
        )

        # Load weights
        instance.load_pretrained(repo_id)
        return instance

    def load_pretrained(self, repo_id: str):
        # Check if repo has PEFT adapter
        try:
            self.model = AutoModelForCausalLM.from_pretrained(repo_id)
            # Try to load as PEFT model
            self.model = PeftModel.from_pretrained(self.model, repo_id)
            self.use_lora = True
        except Exception as e:
            # If it fails, load as regular model
            self.model = AutoModelForCausalLM.from_pretrained(repo_id)
            self.use_lora = False

        # Download safetensors file
        tokens_path = snapshot_download(
            repo_id=repo_id, allow_patterns=["latent_tokens.safetensors"]
        )
        tokens_path = os.path.join(tokens_path, "latent_tokens.safetensors")

        # Load tensors using safetensors
        if os.path.exists(tokens_path):
            tensors = load_file(tokens_path)
            self.gist_tokens.data = tensors["gist_tokens"]
            self.ae_tokens.data = tensors["ae_tokens"]
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
            position_ids=position_ids,
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
        mem_embeds: torch.Tensor,
        input_ids: torch.Tensor,
        max_new_tokens: int,
        temperature: float = 0.01,
        **kwargs,
    ) -> torch.Tensor:
        """Generate text using a more efficient approach with temperature sampling."""
        B = mem_embeds.size(0)
        device = self.model.device

        # Fix for empty input_ids: ensure generated_ids is properly initialized
        if input_ids is None:
            generated_ids = torch.zeros((B, 0), dtype=torch.long, device=device)
        else:
            generated_ids = input_ids.clone()

        mem_embeds = mem_embeds.to(dtype=self.model.dtype)
        if input_ids is not None:
            # Initial input_embeds with memory embeddings
            embeds = self.model.get_input_embeddings()(input_ids)
            # Ensure mem_embeds has the same dtype as embeds
            embeds = torch.cat([mem_embeds, embeds], dim=1)
        else:
            embeds = mem_embeds

        # Create attention mask (1 for all tokens)
        attention_mask = torch.ones(
            (B, embeds.size(1)), dtype=torch.long, device=device
        )
        position_ids = self.position_ids[: embeds.size(1)].repeat(B, 1)
        total_tokens = min(
            max_new_tokens, self.block_size - embeds.size(1) + mem_embeds.size(1)
        )
        # Generate tokens one by one
        for _ in range(total_tokens):
            # Forward pass
            outputs = self.model(
                inputs_embeds=embeds,
                attention_mask=attention_mask,
                position_ids=position_ids,
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
