from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple
import os
from huggingface_hub import HfApi
import numpy as np

CKPT_DIR = ".training_cache/checkpoints"

if not os.path.exists(CKPT_DIR):
    os.makedirs(CKPT_DIR)


class LatentEncoder(nn.Module):
    def __init__(self, model_name: str, n_gist_tokens: int):
        super().__init__()
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
        )
        self.base_config = self.model.config
        self.gist_tokens = nn.Parameter(
            torch.randn(n_gist_tokens, self.base_config.hidden_size)
        )
        self.n_gist_tokens = n_gist_tokens
        self.init_weights()

    def init_weights(self):
        torch.nn.init.kaiming_normal_(self.gist_tokens)

    def get_trainable_parameters(self):
        return [self.gist_tokens, *self.model.parameters()]

    def forward(
        self, input_ids: torch.Tensor, pad_token_id: int
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        B = input_ids.size(0)
        embeds = self.model.get_input_embeddings()(input_ids)
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
        ).hidden_states[-1]
        return last_hidden_states[:, -self.n_gist_tokens :, :]

    def push_to_hub(self, repo_id: str, ckpt_dir: str = CKPT_DIR):
        self.model.push_to_hub(repo_id)
        folder = os.path.dirname(f"{ckpt_dir}/{repo_id}/gist_tokens.npy")
        if not os.path.exists(folder):
            os.makedirs(folder)
        self.gist_tokens.data.cpu().numpy().tofile(
            f"{ckpt_dir}/{repo_id}/gist_tokens.npy"
        )
        hf_api = HfApi()
        np.save(
            f"{ckpt_dir}/{repo_id}/gist_tokens.npy", self.gist_tokens.data.cpu().numpy()
        )
        hf_api.upload_file(
            repo_id=repo_id,
            path_or_fileobj=f"{ckpt_dir}/{repo_id}/gist_tokens.npy",
            path_in_repo="gist_tokens.npy",
        )

    def load_pretrained(self, repo_id: str):
        self.model = AutoModelForCausalLM.from_pretrained(repo_id)
        hf_api = HfApi()
        gist_tokens = hf_api.download_file(
            repo_id=repo_id,
            path_in_repo="gist_tokens.npy",
        )
        self.gist_tokens.data = torch.from_numpy(np.load(gist_tokens))


class LatentDecoder(nn.Module):
    def __init__(self, model_name: str, n_gist_tokens: int):
        super().__init__()
        self.model = AutoModelForCausalLM.from_pretrained(model_name)
        self.base_config = self.model.config
        self.n_gist_tokens = n_gist_tokens

    def forward(
        self,
        input_ids: torch.Tensor,
        mem_embeds: torch.Tensor,
        labels: Optional[torch.Tensor] = None,
        ignore_index: int = -100,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        B, T = input_ids.size()
        assert mem_embeds.size(1) == self.n_gist_tokens
        embeds = self.model.get_input_embeddings()(input_ids)
        embeds = torch.cat(
            [
                mem_embeds,
                embeds,
            ],
            dim=1,
        )
        logits = self.model(inputs_embeds=embeds).logits
        logits = logits[:, self.n_gist_tokens :, :]
        if labels is not None:
            loss = F.cross_entropy(
                logits.reshape(-1, logits.size(-1)),
                labels.reshape(-1),
                ignore_index=ignore_index,
            )
            return logits, loss

        return logits, None

    def generate(
        self,
        mem_embeds: torch.Tensor,
        input_ids: torch.Tensor,
        max_new_tokens: int,
        **kwargs,
    ) -> torch.Tensor:
        generated_ids = input_ids
        embeds = self.model.get_input_embeddings()(input_ids)
        embeds = torch.cat([mem_embeds, embeds], dim=1)
        for _ in range(max_new_tokens):
            logits = self.model(inputs_embeds=embeds).logits
            logits = logits[:, -1, :]
            next_token = torch.argmax(logits, dim=-1)
            if next_token == self.model.config.eos_token_id:
                break
            generated_ids = torch.cat([generated_ids, next_token.unsqueeze(-1)], dim=-1)
            generated_embeds = self.model.get_input_embeddings()(generated_ids)
            embeds = torch.cat([mem_embeds, generated_embeds], dim=1)
        return generated_ids

    def push_to_hub(self, repo_id: str):
        self.model.push_to_hub(repo_id)


if __name__ == "__main__":
    model_name = "HuggingFaceTB/SmolLM2-135M"
    n_gist_tokens = 64
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    latent_encoder = LatentEncoder(model_name, n_gist_tokens)
    latent_decoder = LatentDecoder(model_name, n_gist_tokens)

    input_ids = torch.randint(0, 100, (1, 10))
    mem_embeds = latent_encoder(input_ids)
    logits, loss = latent_decoder(input_ids, mem_embeds, labels=input_ids)
    print(loss)

    completion = latent_decoder.generate(mem_embeds, input_ids, max_new_tokens=10)
    print(tokenizer.decode(completion[0]))
