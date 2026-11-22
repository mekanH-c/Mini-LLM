"""Key-Value (KV) Cache implementation for autoregressive inference."""

from typing import List, Optional, Tuple
import torch

from mini_infer.config import LlamaConfig


class KVCache:
    """Pre-allocated Key-Value Cache for autoregressive generation.

    In naive autoregressive decoding, every new token recomputes key and value projections
    for all previous tokens in the sequence ($O(N^2)$ FLOPs). The KV Cache stores past
    keys and values across all transformer layers so each decode step only computes
    projections for the single newest token ($O(N)$ FLOPs per step).

    Tensor Shapes:
        Each layer stores:
        - k_cache: (batch_size, num_kv_heads, max_seq_len, head_dim)
        - v_cache: (batch_size, num_kv_heads, max_seq_len, head_dim)

    Memory Formula:
        Bytes = 2 * num_layers * batch_size * num_kv_heads * max_seq_len * head_dim * bytes_per_elem
        For SmolLM-135M (L=30, H_KV=3, d=64, S=2048, float32):
        Bytes = 2 * 30 * 1 * 3 * 2048 * 64 * 4 = 94,371,840 bytes (~94.37 MB)

    Args:
        num_layers: Number of transformer blocks (L).
        batch_size: Batch size for inference.
        num_kv_heads: Number of key-value attention heads (H_KV).
        head_dim: Dimension of each attention head.
        max_seq_len: Pre-allocated maximum sequence capacity.
        device: Device to place cache tensors on (default: CPU).
        dtype: Data type of cached tensors (default: torch.float32).
    """

    def __init__(
        self,
        num_layers: int,
        batch_size: int,
        num_kv_heads: int,
        head_dim: int,
        max_seq_len: int = 2048,
        device: Optional[torch.device] = None,
        dtype: torch.dtype = torch.float32,
    ) -> None:
        self.num_layers = num_layers
        self.batch_size = batch_size
        self.num_kv_heads = num_kv_heads
        self.head_dim = head_dim
        self.max_seq_len = max_seq_len
        self.device = device or torch.device("cpu")
        self.dtype = dtype

        # Pre-allocate contiguous zeroed buffers for all layers
        self.k_cache: List[torch.Tensor] = [
            torch.zeros(
                (batch_size, num_kv_heads, max_seq_len, head_dim),
                dtype=dtype,
                device=self.device,
            )
            for _ in range(num_layers)
        ]
        self.v_cache: List[torch.Tensor] = [
            torch.zeros(
                (batch_size, num_kv_heads, max_seq_len, head_dim),
                dtype=dtype,
                device=self.device,
            )
            for _ in range(num_layers)
        ]

    @classmethod
    def from_config(
        cls,
        config: LlamaConfig,
        batch_size: int = 1,
        max_seq_len: int = 2048,
        device: Optional[torch.device] = None,
        dtype: torch.dtype = torch.float32,
    ) -> "KVCache":
        """Convenience constructor from a LlamaConfig instance."""
        return cls(
            num_layers=config.num_hidden_layers,
            batch_size=batch_size,
            num_kv_heads=config.num_key_value_heads,
            head_dim=config.head_dim,
            max_seq_len=max_seq_len,
            device=device,
            dtype=dtype,
        )

    def _grow_cache(self, new_max_len: int) -> None:
        """Dynamically expands the pre-allocated cache buffers if capacity is exceeded."""
        for i in range(self.num_layers):
            new_k = torch.zeros(
                (self.batch_size, self.num_kv_heads, new_max_len, self.head_dim),
                dtype=self.dtype,
                device=self.device,
            )
            new_v = torch.zeros(
                (self.batch_size, self.num_kv_heads, new_max_len, self.head_dim),
                dtype=self.dtype,
                device=self.device,
            )
            new_k[:, :, : self.max_seq_len, :] = self.k_cache[i]
            new_v[:, :, : self.max_seq_len, :] = self.v_cache[i]
            self.k_cache[i] = new_k
            self.v_cache[i] = new_v
        self.max_seq_len = new_max_len

    def update(
        self,
        layer_idx: int,
        k_val: torch.Tensor,
        v_val: torch.Tensor,
        start_pos: int,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Inserts new key and value slices into the cache at `start_pos` and returns valid past.

        Args:
            layer_idx: Index of the calling transformer layer [0, num_layers - 1].
            k_val: Incoming key tensor of shape (batch_size, num_kv_heads, seq_len, head_dim).
            v_val: Incoming value tensor of shape (batch_size, num_kv_heads, seq_len, head_dim).
            start_pos: Starting position index in the sequence (0 for prefill, t for decode).

        Returns:
            Tuple of (keys, values) containing all valid historical tokens up to
            `start_pos + seq_len`, each of shape:
            (batch_size, num_kv_heads, start_pos + seq_len, head_dim)
        """
        batch_size, _, seq_len, _ = k_val.shape
        end_pos = start_pos + seq_len

        # Expand if incoming tokens exceed current capacity
        if end_pos > self.max_seq_len:
            self._grow_cache(max(end_pos, self.max_seq_len * 2))

        # In-place write into pre-allocated memory buffer (zero-copy)
        self.k_cache[layer_idx][:batch_size, :, start_pos:end_pos, :] = k_val
        self.v_cache[layer_idx][:batch_size, :, start_pos:end_pos, :] = v_val

        # Return view of valid keys and values up to end_pos
        return (
            self.k_cache[layer_idx][:batch_size, :, :end_pos, :],
            self.v_cache[layer_idx][:batch_size, :, :end_pos, :],
        )

    def get_allocated_bytes(self) -> int:
        """Returns total allocated buffer memory across all layers in bytes."""
        total = 0
        for k, v in zip(self.k_cache, self.v_cache):
            total += k.numel() * k.element_size()
            total += v.numel() * v.element_size()
        return total

    def get_active_bytes(self, seq_len: int) -> int:
        """Returns the memory occupied by active tokens up to `seq_len` in bytes."""
        elem_size = self.k_cache[0].element_size()
        return (
            2
            * self.num_layers
            * self.batch_size
            * self.num_kv_heads
            * seq_len
            * self.head_dim
            * elem_size
        )

    @staticmethod
    def theoretical_bytes(
        num_layers: int,
        num_kv_heads: int,
        head_dim: int,
        seq_len: int,
        batch_size: int = 1,
        element_size: int = 4,
    ) -> int:
        """Theoretical formula: 2 * L * B * H_KV * S * d_head * element_size."""
        return 2 * num_layers * batch_size * num_kv_heads * seq_len * head_dim * element_size
