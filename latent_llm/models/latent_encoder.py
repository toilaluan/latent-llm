import torch
import torch.nn as nn
import torch.nn.functional as F
import einops
import math


# Token Embedding Module
class TokenEmbedding(nn.Module):
    def __init__(self, embed_dim: int, vocab_size: int):
        """Initialize token embedding with Kaiming normal initialization."""
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        nn.init.kaiming_normal_(self.embedding.weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Embed input tokens."""
        return self.embedding(x.long())


# Positional Embedding Module
class AbsolutePositionalEmbedding(nn.Module):
    def __init__(self, block_size: int, embed_dim: int):
        """Initialize learned positional embeddings with Kaiming normal initialization."""
        super().__init__()
        self.block_size = block_size
        self.weight = nn.Parameter(torch.zeros(block_size, embed_dim))
        nn.init.kaiming_normal_(self.weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Add positional embeddings to input tensor."""
        seq_len = x.size(1)
        batch_size = x.size(0)
        assert (
            seq_len <= self.block_size
        ), f"Sequence length {seq_len} exceeds block size {self.block_size}"
        return self.weight[:seq_len].unsqueeze(0).repeat(batch_size, 1, 1)


# Multi-Head Attention Module
class Attention(nn.Module):
    def __init__(self, embed_dim: int, num_heads: int):
        """Initialize attention mechanism without fixed block size."""
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        assert (
            embed_dim % num_heads == 0
        ), f"Embed dim {embed_dim} not divisible by num heads {num_heads}"
        self.head_dim = embed_dim // num_heads

        self.q_proj = nn.Linear(embed_dim, embed_dim, bias=False)
        self.k_proj = nn.Linear(embed_dim, embed_dim, bias=False)
        self.v_proj = nn.Linear(embed_dim, embed_dim, bias=False)
        self.out_proj = nn.Linear(embed_dim, embed_dim, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Compute causal multi-head attention with dynamic mask."""
        B, T, C = x.size()
        q = self.q_proj(x)
        k = self.k_proj(x)
        v = self.v_proj(x)

        # Reshape for multi-head attention
        q = einops.rearrange(
            q, "b t (h d) -> b h t d", h=self.num_heads, d=self.head_dim
        )
        k = einops.rearrange(
            k, "b t (h d) -> b h t d", h=self.num_heads, d=self.head_dim
        )
        v = einops.rearrange(
            v, "b t (h d) -> b h t d", h=self.num_heads, d=self.head_dim
        )

        # Compute attention weights
        attn_weights = q @ k.transpose(-2, -1) / math.sqrt(self.head_dim)

        # Dynamic causal mask
        mask = torch.tril(torch.ones(T, T, device=x.device)).view(1, 1, T, T)
        attn_weights = attn_weights.masked_fill(mask == 0, float("-inf"))
        attn_weights = F.softmax(attn_weights, dim=-1)

        # Apply attention to values
        out = attn_weights @ v
        out = einops.rearrange(out, "b h t d -> b t (h d)")
        return self.out_proj(out)


# Feed-Forward Network Module
class MLP(nn.Module):
    def __init__(self, embed_dim: int, intermediate_dim: int):
        """Initialize simplified MLP without internal layer norms."""
        super().__init__()
        self.fc1 = nn.Linear(embed_dim, intermediate_dim)
        self.act = nn.GELU(approximate="tanh")
        self.fc2 = nn.Linear(intermediate_dim, embed_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply feed-forward transformations."""
        return self.fc2(self.act(self.fc1(x)))


# Transformer Block
class Block(nn.Module):
    def __init__(self, embed_dim: int, num_heads: int):
        """Initialize transformer block with pre-layer normalization."""
        super().__init__()
        self.ln_1 = nn.LayerNorm(embed_dim)
        self.attn = Attention(embed_dim, num_heads)
        self.ln_2 = nn.LayerNorm(embed_dim)
        self.mlp = MLP(embed_dim, 4 * embed_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Process input through attention and MLP with residual connections."""
        x = x + self.attn(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x


# Encoder Transformer
class EncoderTransformer(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        embed_dim: int,
        num_heads: int,
        block_size: int,
        num_layers: int,
        mem_size: int,
    ):
        """Initialize encoder with memory embeddings."""
        super().__init__()
        self.token_embedding = TokenEmbedding(embed_dim, vocab_size)
        self.positional_embedding = AbsolutePositionalEmbedding(block_size, embed_dim)
        self.blocks = nn.ModuleList(
            [Block(embed_dim, num_heads) for _ in range(num_layers)]
        )
        self.ln_f = nn.LayerNorm(embed_dim)
        self.mem_embeds = nn.Parameter(torch.zeros(mem_size, embed_dim))
        self.mem_ln = nn.LayerNorm(embed_dim)
        self.mem_size = mem_size

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Process input and return last mem_size tokens as memory embeddings."""
        B, T = x.size()
        x = self.token_embedding(x) + self.positional_embedding(x)
        # Prepend initial memory embeddings
        x_mem = torch.cat([self.mem_embeds.repeat(B, 1, 1), x], dim=1)
        for block in self.blocks:
            x_mem = block(x_mem)
        x_mem = self.mem_ln(x_mem)
        # Select last mem_size tokens as output memory
        return x_mem[:, -self.mem_size :, :]


# Decoder Transformer
class DecoderTransformer(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        embed_dim: int,
        num_heads: int,
        block_size: int,
        num_layers: int,
    ):
        """Initialize decoder for language modeling."""
        super().__init__()
        self.token_embedding = TokenEmbedding(embed_dim, vocab_size)
        self.positional_embedding = AbsolutePositionalEmbedding(block_size, embed_dim)
        self.blocks = nn.ModuleList(
            [Block(embed_dim, num_heads) for _ in range(num_layers)]
        )
        self.ln_f = nn.LayerNorm(embed_dim)
        self.lm_head = nn.Linear(embed_dim, vocab_size)

    def forward(
        self,
        x: torch.Tensor,
        mem_embeds: torch.Tensor,
        labels: torch.Tensor,
        ignore_index: int,
    ) -> torch.Tensor:
        """Compute loss using memory embeddings and input sequence."""
        B, T = x.size()
        x = self.token_embedding(x) + self.positional_embedding(x)
        # Prepend memory embeddings from encoder
        x_mem = torch.cat([mem_embeds, x], dim=1)
        for block in self.blocks:
            x_mem = block(x_mem)
        # Extract sequence part after memory
        x = x_mem[:, mem_embeds.size(1) :, :]
        x = self.ln_f(x)
        logits = self.lm_head(x)
        if labels is not None:
            loss = F.cross_entropy(
                logits.reshape(-1, logits.size(-1)),
                labels.reshape(-1),
                ignore_index=ignore_index,
            )
            return logits, loss
        else:
            return logits, None

    def generate(
        self,
        mem_embeds: torch.Tensor,
        start_tokens: torch.Tensor,
        max_length: int,
        eos_token: int,
        top_k: int = 50,
    ):
        """Generate text autoregressively using top-k sampling.

        Args:
            mem_embeds: Memory embeddings from encoder [B, mem_size, D]
            start_tokens: Initial token ids [B, S]
            max_length: Maximum number of tokens to generate
            eos_token: Token ID that signals end of generation
            top_k: Number of highest probability tokens to sample from

        Returns:
            Generated token ids [B, S+new_tokens]
        """
        generated_ids = start_tokens
        B = generated_ids.size(0)

        for _ in range(max_length):
            # Get logits for next token
            logits, _ = self.forward(generated_ids, mem_embeds, None, None)

            # Focus only on the last token's logits for each sequence
            next_token_logits = logits[:, -1, :]  # [B, vocab_size]

            # Apply top-k sampling
            top_k_values, top_k_indices = torch.topk(
                next_token_logits, k=top_k, dim=-1
            )  # [B, top_k]

            # Convert logits to probabilities
            probs = F.softmax(top_k_values, dim=-1)  # [B, top_k]

            # Sample from the top-k distribution
            next_token_indices = torch.multinomial(probs, num_samples=1)  # [B, 1]
            next_tokens = torch.gather(top_k_indices, -1, next_token_indices)  # [B, 1]

            # Append generated token to sequence
            generated_ids = torch.cat([generated_ids, next_tokens], dim=1)

            # Check if any sequence has generated EOS token
            if (next_tokens == eos_token).any():
                break

        return generated_ids


# Example Usage
if __name__ == "__main__":
    import tiktoken

    tokenizer = tiktoken.get_encoding("gpt2")
    vocab_size = tokenizer.n_vocab
    block_size = 1024
    embed_dim = 384
    num_heads = 6
    num_layers = 6
    mem_size = 128

    encoder = EncoderTransformer(
        vocab_size, embed_dim, num_heads, block_size, num_layers, mem_size
    )
    decoder = DecoderTransformer(
        vocab_size, embed_dim, num_heads, block_size, num_layers
    )

    x = torch.randint(0, vocab_size, (1, 256))
    mem_embeds = encoder(x)
    print(mem_embeds.shape)  # Expected: torch.Size([1, 128, 384])
    labels = x[:, 1:]
    x = x[:, :-1]
    loss = decoder(x, mem_embeds, labels)
    print(loss)
    print(mem_embeds.shape)
    generated = decoder.generate(mem_embeds, x[:, :1], 100, tokenizer.encode("<eos>"))
    print(tokenizer.decode(generated[0].tolist()))
