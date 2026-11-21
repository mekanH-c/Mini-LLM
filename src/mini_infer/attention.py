"""Grouped-Query Attention (GQA) with Rotary Position Embeddings for LLaMA architectures."""

import math
from typing import Optional
import torch
import torch.nn as nn
import torch.nn.functional as F

from mini_infer.config import LlamaConfig
from mini_infer.rope import RotaryEmbedding, apply_rotary_pos_emb


def repeat_kv(hidden_states: torch.Tensor, n_rep: int) -> torch.Tensor:
    """Expands Key or Value tensor along the head dimension for Grouped-Query Attention.

    In GQA, multiple query heads share a single key/value head. To perform standard
    batched matrix multiplication, each KV head is repeated `n_rep` times.

    Equivalent to:
        torch.repeat_interleave(hidden_states, repeats=n_rep, dim=1)

    Args:
        hidden_states: Tensor of shape (batch, num_kv_heads, seq_len, head_dim)
        n_rep: Replication factor (num_attention_heads // num_key_value_heads)

    Returns:
        Tensor of shape (batch, num_attention_heads, seq_len, head_dim)
    """
    if n_rep == 1:
        return hidden_states

    batch, num_kv_heads, slen, head_dim = hidden_states.shape
    hidden_states = hidden_states[:, :, None, :, :].expand(
        batch, num_kv_heads, n_rep, slen, head_dim
    )
    return hidden_states.reshape(batch, num_kv_heads * n_rep, slen, head_dim)


class GroupedQueryAttention(nn.Module):
    """Grouped-Query Attention (GQA) module without KV cache (Phase 0-1 naive version).

    In Multi-Head Attention (MHA), H_Q == H_KV.
    In Multi-Query Attention (MQA), H_KV == 1.
    In Grouped-Query Attention (GQA), 1 < H_KV < H_Q. Each KV head is shared by
    (H_Q // H_KV) query heads, reducing KV memory footprint while preserving MHA quality.

    Args:
        config: LlamaConfig containing model hyperparameters.
    """

    def __init__(self, config: LlamaConfig) -> None:
        super().__init__()
        self.hidden_size = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.num_kv_heads = config.num_key_value_heads
        self.head_dim = config.head_dim
        self.num_key_value_groups = self.num_heads // self.num_kv_heads

        # Linear projections without bias
        self.q_proj = nn.Linear(self.hidden_size, self.num_heads * self.head_dim, bias=False)
        self.k_proj = nn.Linear(self.hidden_size, self.num_kv_heads * self.head_dim, bias=False)
        self.v_proj = nn.Linear(self.hidden_size, self.num_kv_heads * self.head_dim, bias=False)
        self.o_proj = nn.Linear(self.num_heads * self.head_dim, self.hidden_size, bias=False)

        # Rotary position embedding
        self.rotary_emb = RotaryEmbedding(
            dim=self.head_dim,
            max_position_embeddings=config.max_position_embeddings,
            base=config.rope_theta,
        )

    def forward(
        self,
        x: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Forward pass for GroupedQueryAttention.

        Args:
            x: Input tensor of shape (batch_size, seq_len, hidden_size)
            attention_mask: Optional additive attention mask broadcastable to
                            (batch_size, 1, seq_len, seq_len). If None, standard
                            causal masking is applied.

        Returns:
            Output tensor of shape (batch_size, seq_len, hidden_size)
        """
        batch_size, seq_len, _ = x.shape

        # Step 1: Linear projections
        # q: (batch_size, seq_len, num_heads * head_dim)
        # k: (batch_size, seq_len, num_kv_heads * head_dim)
        # v: (batch_size, seq_len, num_kv_heads * head_dim)
        q = self.q_proj(x)
        k = self.k_proj(x)
        v = self.v_proj(x)

        # Step 2: Separate heads and transpose for multi-head attention
        # q: (batch_size, num_heads, seq_len, head_dim)
        # k: (batch_size, num_kv_heads, seq_len, head_dim)
        # v: (batch_size, num_kv_heads, seq_len, head_dim)
        q = q.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(batch_size, seq_len, self.num_kv_heads, self.head_dim).transpose(1, 2)
        v = v.view(batch_size, seq_len, self.num_kv_heads, self.head_dim).transpose(1, 2)

        # Step 3: Apply Rotary Position Embeddings (RoPE)
        cos, sin = self.rotary_emb(v, seq_len=seq_len)
        q, k = apply_rotary_pos_emb(q, k, cos, sin)

        # Step 4: Repeat K and V heads to match the number of Q heads (GQA broadcast)
        # k, v: (batch_size, num_heads, seq_len, head_dim)
        k = repeat_kv(k, self.num_key_value_groups)
        v = repeat_kv(v, self.num_key_value_groups)

        # Step 5: Scaled dot-product attention scores
        # scores: (batch_size, num_heads, seq_len, seq_len)
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim)

        # Step 6: Apply causal mask (tokens cannot attend to future positions)
        if attention_mask is not None:
            scores = scores + attention_mask
        elif seq_len > 1:
            causal_mask = torch.triu(
                torch.full(
                    (seq_len, seq_len),
                    float("-inf"),
                    device=scores.device,
                    dtype=scores.dtype,
                ),
                diagonal=1,
            )
            scores = scores + causal_mask[None, None, :, :]

        # Step 7: Softmax over key sequence length in float32 for numerical stability
        attn_weights = F.softmax(scores, dim=-1, dtype=torch.float32).to(q.dtype)

        # Step 8: Compute weighted sum of values
        # context: (batch_size, num_heads, seq_len, head_dim)
        context = torch.matmul(attn_weights, v)

        # Step 9: Concatenate heads and project out
        # Permute: (batch_size, seq_len, num_heads, head_dim)
        # Flatten: (batch_size, seq_len, hidden_size)
        context = context.transpose(1, 2).contiguous().view(batch_size, seq_len, self.hidden_size)

        # Step 10: Final linear output projection
        return self.o_proj(context)
