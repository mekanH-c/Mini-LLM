"""Model configuration for LLaMA architectures."""

from dataclasses import dataclass, field
from typing import Any, Dict, Optional
import json


@dataclass
class LlamaConfig:
    """Hyperparameters for LLaMA-style causal language models.
    
    Default values match SmolLM-135M (HuggingFaceTB/SmolLM-135M).
    """
    vocab_size: int = 49152
    hidden_size: int = 576
    intermediate_size: int = 1536
    num_hidden_layers: int = 30
    num_attention_heads: int = 9
    num_key_value_heads: int = 3
    head_dim: Optional[int] = None
    rms_norm_eps: float = 1e-5
    rope_theta: float = 10000.0
    max_position_embeddings: int = 2048
    tie_word_embeddings: bool = True
    initializer_range: float = 0.02
    extra_kwargs: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.head_dim is None:
            self.head_dim = self.hidden_size // self.num_attention_heads

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "LlamaConfig":
        """Construct LlamaConfig from a configuration dictionary (e.g. HF config.json)."""
        valid_keys = {
            "vocab_size",
            "hidden_size",
            "intermediate_size",
            "num_hidden_layers",
            "num_attention_heads",
            "num_key_value_heads",
            "head_dim",
            "rms_norm_eps",
            "rope_theta",
            "max_position_embeddings",
            "tie_word_embeddings",
            "initializer_range",
        }
        kwargs = {}
        extras = {}
        for k, v in data.items():
            if k in valid_keys:
                kwargs[k] = v
            else:
                extras[k] = v
        return cls(**kwargs, extra_kwargs=extras)

    @classmethod
    def from_json_file(cls, json_path: str) -> "LlamaConfig":
        """Load LlamaConfig from a local JSON file path."""
        with open(json_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return cls.from_dict(data)
