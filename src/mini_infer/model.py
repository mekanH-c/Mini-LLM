"""LLaMA Model Architecture (Transformer Block, Base Model, and Causal LM) with KV Cache support."""

from typing import TYPE_CHECKING, Optional
import torch
import torch.nn as nn

from mini_infer.config import LlamaConfig
from mini_infer.norm import RMSNorm
from mini_infer.attention import GroupedQueryAttention
from mini_infer.mlp import SwiGLUMLP

if TYPE_CHECKING:
    from mini_infer.cache import KVCache


class TransformerBlock(nn.Module):
    """A single Transformer Decoder layer with Pre-RMSNorm, GQA (with KV Cache), and residual connections.

    Flow:
        x -> RMSNorm -> Attention(kv_cache) -> + Residual -> RMSNorm -> MLP -> + Residual -> out

    Args:
        config: LlamaConfig containing layer dimensions.
    """

    def __init__(self, config: LlamaConfig) -> None:
        super().__init__()
        self.input_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.self_attn = GroupedQueryAttention(config)
        self.post_attention_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.mlp = SwiGLUMLP(config)

    def forward(
        self,
        x: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        kv_cache: Optional["KVCache"] = None,
        layer_idx: Optional[int] = None,
        start_pos: int = 0,
    ) -> torch.Tensor:
        """Forward pass for TransformerBlock.

        Args:
            x: Input tensor of shape (batch_size, seq_len, hidden_size)
            attention_mask: Optional attention mask tensor
            kv_cache: Optional KVCache instance for caching keys and values
            layer_idx: Current layer index (0 to num_layers - 1)
            start_pos: Starting position index in the sequence

        Returns:
            Output tensor of shape (batch_size, seq_len, hidden_size)
        """
        # Self-attention with pre-normalization and residual connection
        residual = x
        x = self.input_layernorm(x)
        x = self.self_attn(
            x,
            attention_mask=attention_mask,
            kv_cache=kv_cache,
            layer_idx=layer_idx,
            start_pos=start_pos,
        )
        x = residual + x

        # Feed-forward MLP with pre-normalization and residual connection
        residual = x
        x = self.post_attention_layernorm(x)
        x = self.mlp(x)
        x = residual + x

        return x


class LlamaModel(nn.Module):
    """The core LLaMA Transformer trunk without language modeling head.

    Args:
        config: LlamaConfig configuration object.
    """

    def __init__(self, config: LlamaConfig) -> None:
        super().__init__()
        self.config = config
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size)
        self.layers = nn.ModuleList([
            TransformerBlock(config) for _ in range(config.num_hidden_layers)
        ])
        self.norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        kv_cache: Optional["KVCache"] = None,
        start_pos: int = 0,
    ) -> torch.Tensor:
        """Forward pass through token embeddings, all transformer layers, and final norm.

        Args:
            input_ids: LongTensor of shape (batch_size, seq_len)
            attention_mask: Optional attention mask tensor
            kv_cache: Optional KVCache instance
            start_pos: Starting sequence position (0 for prefill, t for decode)

        Returns:
            hidden_states: FloatTensor of shape (batch_size, seq_len, hidden_size)
        """
        # Shape: (batch_size, seq_len, hidden_size)
        hidden_states = self.embed_tokens(input_ids)

        for layer_idx, layer in enumerate(self.layers):
            hidden_states = layer(
                hidden_states,
                attention_mask=attention_mask,
                kv_cache=kv_cache,
                layer_idx=layer_idx,
                start_pos=start_pos,
            )

        # Final RMSNorm across hidden dimension
        hidden_states = self.norm(hidden_states)
        return hidden_states


class LlamaForCausalLM(nn.Module):
    """LLaMA Model with a causal language modeling head for next-token prediction.

    Args:
        config: LlamaConfig configuration object.
    """

    def __init__(self, config: LlamaConfig) -> None:
        super().__init__()
        self.config = config
        self.model = LlamaModel(config)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

        # Tie word embeddings if specified by config
        if config.tie_word_embeddings:
            self.lm_head.weight = self.model.embed_tokens.weight

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        kv_cache: Optional["KVCache"] = None,
        start_pos: int = 0,
    ) -> torch.Tensor:
        """Forward pass to compute unnormalized token logits.

        Args:
            input_ids: LongTensor of shape (batch_size, seq_len)
            attention_mask: Optional attention mask tensor
            kv_cache: Optional KVCache instance
            start_pos: Starting sequence position

        Returns:
            logits: FloatTensor of shape (batch_size, seq_len, vocab_size)
        """
        # hidden_states: (batch_size, seq_len, hidden_size)
        hidden_states = self.model(
            input_ids,
            attention_mask=attention_mask,
            kv_cache=kv_cache,
            start_pos=start_pos,
        )
        # logits: (batch_size, seq_len, vocab_size)
        logits = self.lm_head(hidden_states)
        return logits
