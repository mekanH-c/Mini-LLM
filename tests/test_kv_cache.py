"""Unit tests for KV Cache, RoPE position consistency, and Cached Generation on SmolLM-135M."""

import pytest
import torch
from transformers import AutoTokenizer

from mini_infer.config import LlamaConfig
from mini_infer.rope import RotaryEmbedding, apply_rotary_pos_emb
from mini_infer.cache import KVCache
from mini_infer.attention import GroupedQueryAttention
from mini_infer.loader import load_from_hf
from mini_infer.generate import generate, generate_cached

MODEL_ID = "HuggingFaceTB/SmolLM-135M"


@pytest.fixture(scope="module")
def model_and_tokenizer():
    """Load SmolLM-135M custom model and tokenizer in float32."""
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    model, _ = load_from_hf(MODEL_ID, dtype=torch.float32)
    model.eval()
    return model, tokenizer


def test_rope_decode_position_angles() -> None:
    """Test that a cached decode step at position t gets the same rotary angles as position t in the full run."""
    dim = 64
    max_len = 2048
    rotary = RotaryEmbedding(dim=dim, max_position_embeddings=max_len)
    dummy_x = torch.zeros(1, 1, 1, dim)

    # Test diverse position indices across the context window
    test_positions = [0, 1, 5, 17, 31, 63, 127, 255, 511, 1023, 2047]

    for t in test_positions:
        # Full run: sequence from 0 to t (length t + 1)
        cos_full, sin_full = rotary(dummy_x, seq_len=t + 1, start_pos=0)
        cos_full_t = cos_full[:, :, t : t + 1, :]
        sin_full_t = sin_full[:, :, t : t + 1, :]

        # Cached decode step: 1 token at position t
        cos_decode, sin_decode = rotary(dummy_x, seq_len=1, start_pos=t)

        # 1. Exact angle equality
        assert torch.equal(cos_full_t, cos_decode), f"cos mismatch at position {t}"
        assert torch.equal(sin_full_t, sin_decode), f"sin mismatch at position {t}"

        # 2. Applied rotation equality on a single query token across all 9 heads
        batch_size = 2
        num_heads = 9
        num_kv_heads = 3
        q_single = torch.randn(batch_size, num_heads, 1, dim)
        k_single = torch.randn(batch_size, num_kv_heads, 1, dim)

        # Apply using decode slice
        q_rot_decode, k_rot_decode = apply_rotary_pos_emb(q_single, k_single, cos_decode, sin_decode)

        # Apply using full run slice at index t
        q_rot_full, k_rot_full = apply_rotary_pos_emb(q_single, k_single, cos_full_t, sin_full_t)

        assert torch.allclose(q_rot_decode, q_rot_full, atol=1e-7), f"q rotation divergence at position {t}"
        assert torch.allclose(k_rot_decode, k_rot_full, atol=1e-7), f"k rotation divergence at position {t}"


def test_kv_cache_memory_formula() -> None:
    """Verify KV cache memory matches the theoretical formula: 2 * L * B * H_KV * S * d_head * element_size."""
    config = LlamaConfig()  # SmolLM-135M defaults: L=30, H_KV=3, d=64, S=2048
    cache = KVCache.from_config(config, batch_size=1, max_seq_len=2048, dtype=torch.float32)

    # Theoretical SmolLM-135M float32 allocation:
    # 2 (K and V) * 30 layers * 1 batch * 3 heads * 2048 tokens * 64 dim * 4 bytes
    expected_total_bytes = 2 * 30 * 1 * 3 * 2048 * 64 * 4
    assert expected_total_bytes == 94_371_840  # exactly 90 MiB (~94.37 MB)
    assert cache.get_allocated_bytes() == expected_total_bytes

    # Check active bytes at various sequence lengths
    for s in [1, 32, 128, 512, 2048]:
        expected_active = 2 * 30 * 1 * 3 * s * 64 * 4
        assert cache.get_active_bytes(s) == expected_active
        assert KVCache.theoretical_bytes(30, 3, 64, s, batch_size=1, element_size=4) == expected_active


def test_gqa_cache_vs_naive_numerical_equivalence() -> None:
    """Verify GroupedQueryAttention with KV Cache produces identical outputs to naive full-sequence attention."""
    torch.manual_seed(42)
    config = LlamaConfig(
        hidden_size=64,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=16,
        max_position_embeddings=128,
    )
    attn = GroupedQueryAttention(config)
    attn.eval()

    batch_size = 1
    total_len = 6
    prefill_len = 3
    x_full = torch.randn(batch_size, total_len, config.hidden_size)

    # 1. Naive execution on the full sequence
    with torch.no_grad():
        out_naive = attn(x_full)  # (1, 6, 64)

    # 2. Cached execution: prefill first 3 tokens, then decode tokens 3, 4, 5 one by one
    cache = KVCache(
        num_layers=1,
        batch_size=batch_size,
        num_kv_heads=config.num_key_value_heads,
        head_dim=config.head_dim,
        max_seq_len=128,
    )

    with torch.no_grad():
        # Prefill: positions 0..2
        out_prefill = attn(
            x_full[:, :prefill_len, :],
            kv_cache=cache,
            layer_idx=0,
            start_pos=0,
        )

        assert torch.allclose(out_prefill, out_naive[:, :prefill_len, :], atol=1e-5), (
            f"Prefill mismatch: max diff = {(out_prefill - out_naive[:, :prefill_len, :]).abs().max()}"
        )

        # Decode tokens one by one
        for t in range(prefill_len, total_len):
            x_step = x_full[:, t : t + 1, :]
            out_step = attn(
                x_step,
                kv_cache=cache,
                layer_idx=0,
                start_pos=t,
            )
            naive_step = out_naive[:, t : t + 1, :]
            max_diff = (out_step - naive_step).abs().max().item()
            assert max_diff < 1e-5, f"Decode step {t} mismatch: max diff = {max_diff}"


# Real prompts from test_phase01_correctness.py
REAL_PROMPTS_5 = [
    "The capital of France is",
    "def fibonacci(n):",
    "The primary purpose of an operating system is",
    "To make a delicious cup of coffee, you need",
    "Machine learning models require data to",
]


def test_real_smollm_5_prompts_cached_vs_naive(model_and_tokenizer) -> None:
    """Verify that generate_cached produces identical tokens to naive generate on 5 real prompts (25 tokens each)."""
    model, tokenizer = model_and_tokenizer
    max_new_tokens = 25

    print(f"\n--- Testing 5 Real Prompts: Cached vs Naive ({max_new_tokens} tokens each) ---")

    for i, prompt in enumerate(REAL_PROMPTS_5, 1):
        # 1. Naive generation (full recomputation at every decode step)
        out_naive = generate(
            model=model,
            tokenizer=tokenizer,
            prompts=prompt,
            max_new_tokens=max_new_tokens,
            temperature=0.0,
        )
        naive_tokens = out_naive.token_ids[0].tolist()
        naive_gen_tokens = naive_tokens[out_naive.prompt_tokens :]

        # 2. KV Cached generation (prefill once, then decode 1 token at start_pos = t)
        out_cached = generate_cached(
            model=model,
            tokenizer=tokenizer,
            prompts=prompt,
            max_new_tokens=max_new_tokens,
            temperature=0.0,
        )
        cached_tokens = out_cached.token_ids[0].tolist()
        cached_gen_tokens = cached_tokens[out_cached.prompt_tokens :]

        # Assert token-for-token equality
        assert cached_tokens == naive_tokens, (
            f"Prompt {i} ('{prompt}') mismatch!\n"
            f"Cached IDs: {cached_tokens}\n"
            f"Naive IDs:  {naive_tokens}"
        )

        # Assert output tokens are NOT all identical
        unique_tokens = set(cached_gen_tokens)
        assert len(unique_tokens) > 1, f"Prompt {i} generated degenerate repeated token: {unique_tokens}"

        print(f"\nPrompt {i}/{len(REAL_PROMPTS_5)}: '{prompt}'")
        print(f"  Exact Match:   YES (all {max_new_tokens} tokens)")
        print(f"  Unique Tokens: {len(unique_tokens)} / {len(cached_gen_tokens)}")
        print(f"  Generated Text: {out_cached.text[0]!r}")


def test_real_smollm_long_prompt_cache_depth(model_and_tokenizer) -> None:
    """Verify cached decode after a long prompt (>= 50 tokens) to test deep cache positions."""
    model, tokenizer = model_and_tokenizer

    # 63-token paragraph
    long_prompt = (
        "The history of computing hardware covers the developments from early simple devices to aid "
        "calculation to modern day computers. Before the twentieth century, most calculations were "
        "done by humans. The first aids to computation were purely mechanical devices which required "
        "the operator to set up the initial values and manipulate the mechanism to achieve a result."
    )
    prompt_tokens = tokenizer(long_prompt, return_tensors="pt")["input_ids"]
    prompt_len = prompt_tokens.shape[1]
    assert prompt_len >= 50, f"Prompt length {prompt_len} is less than 50 tokens!"

    max_new_tokens = 25
    print(f"\n--- Testing Long Prompt (Prompt Len: {prompt_len} tokens, Gen: {max_new_tokens} tokens) ---")

    out_naive = generate(
        model=model,
        tokenizer=tokenizer,
        prompts=long_prompt,
        max_new_tokens=max_new_tokens,
        temperature=0.0,
    )

    out_cached = generate_cached(
        model=model,
        tokenizer=tokenizer,
        prompts=long_prompt,
        max_new_tokens=max_new_tokens,
        temperature=0.0,
    )

    naive_ids = out_naive.token_ids[0].tolist()
    cached_ids = out_cached.token_ids[0].tolist()

    assert cached_ids == naive_ids, (
        f"Long prompt mismatch!\nCached: {cached_ids}\nNaive:  {naive_ids}"
    )

    gen_tokens = cached_ids[prompt_len:]
    unique_tokens = set(gen_tokens)
    assert len(unique_tokens) > 1, f"Long prompt produced degenerate repeated token: {unique_tokens}"

    print(f"  Prompt Length: {prompt_len} tokens")
    print(f"  Exact Match:   YES (all {max_new_tokens} tokens match naive)")
    print(f"  Cache Depth:   Successfully cached and retrieved across positions 0 to {prompt_len + max_new_tokens - 1}")
    print(f"  Generated Text: {out_cached.text[0]!r}")
