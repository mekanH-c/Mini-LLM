"""Direct safetensors weight loader for custom LLaMA models."""

import os
import json
from typing import Dict, List, Optional, Tuple
import torch
from safetensors.torch import load_file
from huggingface_hub import hf_hub_download

from mini_infer.config import LlamaConfig
from mini_infer.model import LlamaForCausalLM


def get_hf_key_mapping(config: LlamaConfig) -> Dict[str, str]:
    """Generates the explicit mapping from Hugging Face checkpoint keys to our model's parameter keys.

    Args:
        config: LlamaConfig containing number of layers.

    Returns:
        Dictionary where:
            key = Hugging Face state_dict key name
            value = Our custom LlamaForCausalLM state_dict key name
    """
    mapping: Dict[str, str] = {
        "model.embed_tokens.weight": "model.embed_tokens.weight",
        "model.norm.weight": "model.norm.weight",
    }

    if not config.tie_word_embeddings:
        mapping["lm_head.weight"] = "lm_head.weight"
    else:
        # When tied, lm_head shares weights with embed_tokens
        mapping["lm_head.weight"] = "model.embed_tokens.weight"

    for i in range(config.num_hidden_layers):
        layer_hf = f"model.layers.{i}"
        layer_custom = f"model.layers.{i}"

        # Attention layer keys
        mapping[f"{layer_hf}.input_layernorm.weight"] = f"{layer_custom}.input_layernorm.weight"
        mapping[f"{layer_hf}.self_attn.q_proj.weight"] = f"{layer_custom}.self_attn.q_proj.weight"
        mapping[f"{layer_hf}.self_attn.k_proj.weight"] = f"{layer_custom}.self_attn.k_proj.weight"
        mapping[f"{layer_hf}.self_attn.v_proj.weight"] = f"{layer_custom}.self_attn.v_proj.weight"
        mapping[f"{layer_hf}.self_attn.o_proj.weight"] = f"{layer_custom}.self_attn.o_proj.weight"

        # MLP layer keys
        mapping[f"{layer_hf}.post_attention_layernorm.weight"] = f"{layer_custom}.post_attention_layernorm.weight"
        mapping[f"{layer_hf}.mlp.gate_proj.weight"] = f"{layer_custom}.mlp.gate_proj.weight"
        mapping[f"{layer_hf}.mlp.up_proj.weight"] = f"{layer_custom}.mlp.up_proj.weight"
        mapping[f"{layer_hf}.mlp.down_proj.weight"] = f"{layer_custom}.mlp.down_proj.weight"

    return mapping


def count_parameters(model: torch.nn.Module) -> Dict[str, int]:
    """Counts parameters in a model, distinguishing total and unique (untied) parameters.

    Returns:
        Dictionary with:
            - 'total_parameters': sum of all parameter elements
            - 'unique_parameters': sum of elements excluding tied duplicates
    """
    total_params = sum(p.numel() for p in model.parameters())
    unique_params = sum(
        p.numel() for p in {p.data_ptr(): p for p in model.parameters()}.values()
    )
    return {
        "total_parameters": total_params,
        "unique_parameters": unique_params,
    }


def resolve_model_files(model_name_or_path: str) -> Tuple[str, List[str]]:
    """Resolves local paths to config.json and safetensors files.

    If model_name_or_path is a local directory, finds files on disk.
    If it is a Hugging Face repo ID, downloads them via huggingface_hub.

    Returns:
        Tuple of (config_path, list_of_safetensors_paths)
    """
    if os.path.isdir(model_name_or_path):
        config_path = os.path.join(model_name_or_path, "config.json")
        if not os.path.exists(config_path):
            raise FileNotFoundError(f"config.json not found in {model_name_or_path}")

        safetensors_files = [
            os.path.join(model_name_or_path, f)
            for f in os.listdir(model_name_or_path)
            if f.endswith(".safetensors")
        ]
        if not safetensors_files:
            raise FileNotFoundError(f"No .safetensors files found in {model_name_or_path}")
        return config_path, sorted(safetensors_files)

    # Download from Hugging Face Hub
    config_path = hf_hub_download(repo_id=model_name_or_path, filename="config.json")

    # Check if model has an index file for multiple shards
    try:
        index_path = hf_hub_download(
            repo_id=model_name_or_path,
            filename="model.safetensors.index.json",
        )
        with open(index_path, "r", encoding="utf-8") as f:
            index_data = json.load(f)
        shard_names = sorted(list(set(index_data["weight_map"].values())))
        safetensors_files = [
            hf_hub_download(repo_id=model_name_or_path, filename=name)
            for name in shard_names
        ]
    except Exception:
        # Single safetensors file
        single_path = hf_hub_download(
            repo_id=model_name_or_path,
            filename="model.safetensors",
        )
        safetensors_files = [single_path]

    return config_path, safetensors_files


def load_from_hf(
    model_name_or_path: str,
    device: str = "cpu",
    dtype: torch.dtype = torch.float32,
) -> Tuple[LlamaForCausalLM, LlamaConfig]:
    """Loads a LLaMA model directly from safetensors weights into our custom architecture.

    Args:
        model_name_or_path: Local folder path or Hugging Face Hub repository identifier.
        device: Target torch device ('cpu', 'cuda', etc.).
        dtype: Target torch data type (torch.float32, torch.bfloat16, etc.).

    Returns:
        Tuple of (model, config) where model has weights loaded and is set to eval() mode.
    """
    config_path, safetensors_files = resolve_model_files(model_name_or_path)
    config = LlamaConfig.from_json_file(config_path)

    # Instantiate model without initializing random weights to save memory
    model = LlamaForCausalLM(config).to(device=device, dtype=dtype)

    # Load all weights from safetensors
    state_dict: Dict[str, torch.Tensor] = {}
    for st_path in safetensors_files:
        shard_dict = load_file(st_path, device=device)
        state_dict.update(shard_dict)

    # Build key mapping
    key_mapping = get_hf_key_mapping(config)

    # Populate model parameters
    with torch.no_grad():
        model_state = model.state_dict()
        for hf_key, custom_key in key_mapping.items():
            if hf_key in state_dict:
                source_tensor = state_dict[hf_key].to(device=device, dtype=dtype)
                if custom_key in model_state:
                    if model_state[custom_key].shape != source_tensor.shape:
                        raise ValueError(
                            f"Shape mismatch for key '{hf_key}' -> '{custom_key}': "
                            f"Model expected {model_state[custom_key].shape}, "
                            f"got {source_tensor.shape} from checkpoint."
                        )
                    model_state[custom_key].copy_(source_tensor)

        # Ensure word embeddings and lm_head are tied if configured
        if config.tie_word_embeddings:
            model.lm_head.weight = model.model.embed_tokens.weight

    model.eval()
    return model, config
