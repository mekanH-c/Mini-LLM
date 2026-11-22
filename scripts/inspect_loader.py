"""Script to verify model loading, count parameters, and print HF-to-custom key mappings."""

import sys
import os

# Add src to python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src")))

from mini_infer.loader import load_from_hf, count_parameters, get_hf_key_mapping
from mini_infer.config import LlamaConfig


def main():
    model_name = "HuggingFaceTB/SmolLM-135M"
    print(f"Loading {model_name}...")
    model, config = load_from_hf(model_name)
    print("Model loaded successfully into custom LlamaForCausalLM!")

    # Count parameters
    param_counts = count_parameters(model)
    print("\n--- Parameter Counts ---")
    print(f"Total parameters (including tied): {param_counts['total_parameters']:,}")
    print(f"Unique parameters (untied):        {param_counts['unique_parameters']:,}")

    # Inspect key mapping
    mapping = get_hf_key_mapping(config)
    print(f"\n--- HF-Key to Module Mapping ({len(mapping)} total tensor mappings) ---")
    
    # Print embeddings and norm
    print(f"{'Hugging Face Key':<45} -> {'Custom Module Key':<45}")
    print("-" * 92)
    print(f"{'model.embed_tokens.weight':<45} -> {mapping['model.embed_tokens.weight']:<45}")
    print(f"{'model.norm.weight':<45} -> {mapping['model.norm.weight']:<45}")
    print(f"{'lm_head.weight':<45} -> {mapping['lm_head.weight']:<45} (tied to embed_tokens)")

    # Print first layer and last layer as representatives
    print("\n[Layer 0 Representative]")
    for k, v in mapping.items():
        if "layers.0." in k:
            print(f"{k:<45} -> {v:<45}")

    print("\n[Layer 29 Representative (Last Layer)]")
    for k, v in mapping.items():
        if "layers.29." in k:
            print(f"{k:<45} -> {v:<45}")

    print("\nSample forward test with dummy input_ids:")
    import torch
    dummy_input = torch.tensor([[1, 150, 320, 45]], dtype=torch.long)
    with torch.no_grad():
        logits = model(dummy_input)
    print(f"Input shape:  {dummy_input.shape}")
    print(f"Logits shape: {logits.shape} (batch_size=1, seq_len=4, vocab_size={config.vocab_size})")


if __name__ == "__main__":
    main()
