"""Mini LLM Inference Engine from scratch."""

from mini_infer.config import LlamaConfig
from mini_infer.norm import RMSNorm
from mini_infer.rope import RotaryEmbedding, apply_rotary_pos_emb
from mini_infer.mlp import SwiGLUMLP
from mini_infer.attention import GroupedQueryAttention, repeat_kv
from mini_infer.model import TransformerBlock, LlamaModel, LlamaForCausalLM
from mini_infer.loader import (
    load_from_hf,
    get_hf_key_mapping,
    count_parameters,
    resolve_model_files,
)

__version__ = "0.1.0"

__all__ = [
    "LlamaConfig",
    "RMSNorm",
    "RotaryEmbedding",
    "apply_rotary_pos_emb",
    "SwiGLUMLP",
    "GroupedQueryAttention",
    "repeat_kv",
    "TransformerBlock",
    "LlamaModel",
    "LlamaForCausalLM",
    "load_from_hf",
    "get_hf_key_mapping",
    "count_parameters",
    "resolve_model_files",
]
