"""End-to-end correctness verification of custom LLaMA against Hugging Face SmolLM-135M."""

import pytest
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from mini_infer.loader import load_from_hf
from mini_infer.generate import generate

MODEL_ID = "HuggingFaceTB/SmolLM-135M"


@pytest.fixture(scope="module")
def models_and_tokenizer():
    """Load both custom and Hugging Face reference models with the tokenizer in float32."""
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    hf_model = AutoModelForCausalLM.from_pretrained(MODEL_ID, torch_dtype=torch.float32)
    hf_model.eval()

    custom_model, _ = load_from_hf(MODEL_ID, dtype=torch.float32)
    custom_model.eval()

    return custom_model, hf_model, tokenizer


def test_full_model_logits_match(models_and_tokenizer):
    """Verify that full-model forward pass logits match Hugging Face within 1e-4 tolerance."""
    custom_model, hf_model, tokenizer = models_and_tokenizer

    prompt = "The quick brown fox jumps over the lazy dog."
    encoded = tokenizer(prompt, return_tensors="pt")
    input_ids = encoded["input_ids"]

    with torch.no_grad():
        hf_outputs = hf_model(input_ids)
        hf_logits = hf_outputs.logits

        custom_logits = custom_model(input_ids)

    # Compute maximum absolute and relative differences
    max_abs_diff = (custom_logits - hf_logits).abs().max().item()
    mean_abs_diff = (custom_logits - hf_logits).abs().mean().item()

    print(f"\n--- Logit Comparison Metrics ---")
    print(f"Max Absolute Logit Difference:  {max_abs_diff:.6e}")
    print(f"Mean Absolute Logit Difference: {mean_abs_diff:.6e}")

    assert max_abs_diff < 1e-4, f"Logit max difference {max_abs_diff} exceeds 1e-4"
    torch.testing.assert_close(custom_logits, hf_logits, atol=1e-4, rtol=1e-4)


PROMPTS_10 = [
    "The capital of France is",
    "Once upon a time, in a faraway land,",
    "def fibonacci(n):",
    "In mathematics, the Riemann hypothesis is",
    "The quick brown fox jumps",
    "The primary purpose of an operating system is",
    "To make a delicious cup of coffee, you need",
    "import numpy as np\nimport torch\n",
    "The largest ocean on Earth is",
    "Machine learning models require data to",
]


def test_greedy_generation_identical_10_prompts(models_and_tokenizer):
    """Verify greedy decoding generates 100% token-identical output for 10 diverse prompts."""
    custom_model, hf_model, tokenizer = models_and_tokenizer

    max_new_tokens = 20
    matches = 0
    total = len(PROMPTS_10)

    print(f"\n--- Greedy Generation Comparison (10 Prompts, {max_new_tokens} tokens each) ---")

    for i, prompt in enumerate(PROMPTS_10, 1):
        # 1. Custom generation
        custom_out = generate(
            custom_model,
            tokenizer,
            prompts=prompt,
            max_new_tokens=max_new_tokens,
            temperature=0.0,
        )
        custom_tokens = custom_out.token_ids[0].tolist()
        custom_text = custom_out.text[0]

        # 2. Hugging Face generation
        hf_input = tokenizer(prompt, return_tensors="pt")["input_ids"]
        with torch.no_grad():
            hf_out_tokens = hf_model.generate(
                hf_input,
                max_new_tokens=max_new_tokens,
                do_sample=False,
            )[0].tolist()
        hf_text = tokenizer.decode(hf_out_tokens, skip_special_tokens=True)

        is_match = (custom_tokens == hf_out_tokens)
        if is_match:
            matches += 1

        print(f"\nPrompt {i}/{total}: '{prompt}'")
        print(f"  Match:       {'YES' if is_match else 'NO'}")
        print(f"  Custom Text: {custom_text!r}")
        print(f"  HF Text:     {hf_text!r}")
        if not is_match:
            print(f"  Custom IDs:  {custom_tokens}")
            print(f"  HF IDs:      {hf_out_tokens}")

    print(f"\nSummary: {matches}/{total} prompts matched exactly.")
    assert matches == total, f"Only {matches}/{total} prompts matched HF reference exactly!"
