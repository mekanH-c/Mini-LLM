"""Rotary Position Embeddings (RoPE)."""

from typing import Tuple
import torch
import torch.nn as nn


def rotate_half(x: torch.Tensor) -> torch.Tensor:
    """Rotates half the hidden dimensions of the input tensor.

    Splits the head_dim into two equal halves: [x1, x2] -> [-x2, x1].
    Matches the Hugging Face LLaMA reference implementation convention.

    Args:
        x: Tensor of shape (..., head_dim)

    Returns:
        Tensor of shape (..., head_dim) with halves rotated and negated.
    """
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


def apply_rotary_pos_emb(
    q: torch.Tensor,
    k: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Applies Rotary Position Embedding to query and key tensors.

    Formula:
        R(x) = (x * cos) + (rotate_half(x) * sin)

    Args:
        q: Query tensor of shape (batch_size, num_heads, seq_len, head_dim)
        k: Key tensor of shape (batch_size, num_kv_heads, seq_len, head_dim)
        cos: Cosine frequencies of shape (1, 1, seq_len, head_dim)
        sin: Sine frequencies of shape (1, 1, seq_len, head_dim)

    Returns:
        Tuple of (rotated_q, rotated_k) with identical shapes to inputs.
    """
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed


class RotaryEmbedding(nn.Module):
    """Precomputes rotary position embedding frequencies and trigonometric tables.

    Frequencies formula:
        theta_i = base^(-2(i - 1)/dim) for i in [1, 2, ..., dim/2]

    Args:
        dim: Dimension of each attention head (head_dim).
        max_position_embeddings: Maximum sequence length supported.
        base: Base period for geometric progression of frequencies (default: 10000.0).
    """

    def __init__(
        self,
        dim: int,
        max_position_embeddings: int = 2048,
        base: float = 10000.0,
    ) -> None:
        super().__init__()
        self.dim = dim
        self.max_position_embeddings = max_position_embeddings
        self.base = base

        # Inverse frequencies: theta_i = 1.0 / (base ** (2i / dim))
        inv_freq = 1.0 / (self.base ** (torch.arange(0, self.dim, 2).float() / self.dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)

        # Precompute table up to max_position_embeddings
        self._set_cos_sin_cache(max_position_embeddings)

    def _set_cos_sin_cache(self, seq_len: int) -> None:
        t = torch.arange(seq_len, dtype=torch.float32)
        # Outer product: (seq_len, dim // 2)
        freqs = torch.outer(t, self.inv_freq)
        # Duplicate to form full head_dim: (seq_len, dim)
        emb = torch.cat((freqs, freqs), dim=-1)

        # Cache shaped as (1, 1, seq_len, dim) for broadcasting across (batch, num_heads)
        self.register_buffer("cos_cached", emb.cos()[None, None, :, :], persistent=False)
        self.register_buffer("sin_cached", emb.sin()[None, None, :, :], persistent=False)

    def forward(
        self,
        x: torch.Tensor,
        seq_len: int,
        start_pos: int = 0,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Retrieve precomputed cos and sin slices starting at `start_pos` for length `seq_len`.

        Args:
            x: Input tensor (used to match device and dtype)
            seq_len: Number of tokens in the current sequence chunk (1 during decode)
            start_pos: Starting position index in the sequence (0 for prefill, t for decode)

        Returns:
            Tuple of (cos, sin) tensors of shape (1, 1, seq_len, dim)
        """
        end_pos = start_pos + seq_len
        if end_pos > self.cos_cached.shape[2]:
            self._set_cos_sin_cache(end_pos)

        cos = self.cos_cached[:, :, start_pos:end_pos, :].to(dtype=x.dtype, device=x.device)
        sin = self.sin_cached[:, :, start_pos:end_pos, :].to(dtype=x.dtype, device=x.device)
        return cos, sin
