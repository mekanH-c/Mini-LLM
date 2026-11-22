"""Autoregressive text generation loops (Naive and KV Cached) with performance metrics."""

import time
from dataclasses import dataclass
from typing import Any, List, Optional, Union
import torch

from mini_infer.cache import KVCache
from mini_infer.model import LlamaForCausalLM
from mini_infer.sampling import sample


@dataclass
class GenerationOutput:
    """Stores generated sequences and benchmarking telemetry."""
    token_ids: torch.Tensor
    text: List[str]
    ttft_ms: float
    total_time_ms: float
    prompt_tokens: int
    generated_tokens: int
    tokens_per_second: float
    ms_per_token: float = 0.0


def generate(
    model: LlamaForCausalLM,
    tokenizer: Any,
    prompts: Union[str, List[str]],
    max_new_tokens: int = 32,
    temperature: float = 0.0,
    top_k: Optional[int] = None,
    top_p: Optional[float] = None,
    eos_token_id: Optional[int] = None,
    device: str = "cpu",
    time_budget_s: Optional[float] = None,
) -> GenerationOutput:
    """Generates text autoregressively using naive full-sequence forward passes (no KV cache).

    On each generation step t, the entire sequence (prompt + tokens generated so far)
    is passed through the transformer model:
        x_{1:t} -> Model -> Logits_{t} -> Sample -> x_{t+1}

    Args:
        model: Custom LlamaForCausalLM model.
        tokenizer: Hugging Face tokenizer instance.
        prompts: Single string or list of prompt strings.
        max_new_tokens: Maximum number of tokens to generate per prompt.
        temperature: Sampling temperature (0.0 = greedy).
        top_k: Top-K filtering parameter.
        top_p: Top-P nucleus sampling parameter.
        eos_token_id: End of sequence token ID to terminate generation.
        device: Torch execution device.
        time_budget_s: Optional time budget in seconds. If exceeded, stops early.

    Returns:
        GenerationOutput containing generated token IDs, decoded text, and timing metrics.
    """
    model.eval()

    if isinstance(prompts, str):
        prompts = [prompts]

    # Tokenize prompts
    encoded = tokenizer(prompts, return_tensors="pt")
    input_ids = encoded["input_ids"].to(device)
    prompt_tokens = input_ids.shape[1]

    if eos_token_id is None:
        eos_token_id = getattr(tokenizer, "eos_token_id", None)

    curr_input_ids = input_ids.clone()
    generated_token_list: List[torch.Tensor] = []

    # Benchmark timers
    t_start = time.perf_counter()
    t_first_token: Optional[float] = None

    with torch.no_grad():
        for step in range(max_new_tokens):
            if time_budget_s is not None and (time.perf_counter() - t_start) > time_budget_s:
                break

            # Naive forward pass over entire sequence
            logits = model(curr_input_ids)

            # Extract logits for the final position: (batch_size, vocab_size)
            next_token_logits = logits[:, -1, :]

            # Sample next token: (batch_size, 1)
            next_token = sample(
                next_token_logits,
                temperature=temperature,
                top_k=top_k,
                top_p=top_p,
            )

            if step == 0:
                t_first_token = time.perf_counter()

            generated_token_list.append(next_token)
            curr_input_ids = torch.cat([curr_input_ids, next_token], dim=-1)

            # Check for EOS across all batch items
            if eos_token_id is not None and (next_token == eos_token_id).all():
                break

    t_end = time.perf_counter()

    ttft_ms = (t_first_token - t_start) * 1000.0 if t_first_token is not None else 0.0
    total_time_ms = (t_end - t_start) * 1000.0

    generated_tokens_count = len(generated_token_list)
    decode_tokens_count = max(0, generated_tokens_count - 1)
    decode_time_s = (t_end - t_first_token) if t_first_token is not None else 0.0

    tokens_per_second = (
        (decode_tokens_count / decode_time_s) if decode_time_s > 0 else 0.0
    )
    ms_per_token = (
        (decode_time_s * 1000.0 / decode_tokens_count) if decode_tokens_count > 0 else 0.0
    )

    # Decode text
    decoded_texts = [
        tokenizer.decode(curr_input_ids[i], skip_special_tokens=True)
        for i in range(curr_input_ids.shape[0])
    ]

    return GenerationOutput(
        token_ids=curr_input_ids,
        text=decoded_texts,
        ttft_ms=ttft_ms,
        total_time_ms=total_time_ms,
        prompt_tokens=prompt_tokens,
        generated_tokens=generated_tokens_count,
        tokens_per_second=tokens_per_second,
        ms_per_token=ms_per_token,
    )


def generate_cached(
    model: LlamaForCausalLM,
    tokenizer: Any,
    prompts: Union[str, List[str]],
    max_new_tokens: int = 32,
    temperature: float = 0.0,
    top_k: Optional[int] = None,
    top_p: Optional[float] = None,
    eos_token_id: Optional[int] = None,
    device: str = "cpu",
    max_seq_len: int = 2048,
    time_budget_s: Optional[float] = None,
) -> GenerationOutput:
    """Generates text autoregressively using a Key-Value (KV) Cache.

    Decouples execution into two phases:
    1. Prefill Phase: Prompt tokens are processed in a single forward pass (O(P^2)),
       populating the KV cache with keys and values for positions [0..P-1].
       Logits at position P-1 yield the first generated token.
    2. Decode Phase: For each subsequent step t, only 1 new token is fed into the model
       at start_pos = t (O(P + t) attention complexity), appending its K and V to the cache.

    Args:
        model: Custom LlamaForCausalLM model.
        tokenizer: Hugging Face tokenizer instance.
        prompts: Single string or list of prompt strings.
        max_new_tokens: Maximum number of tokens to generate per prompt.
        temperature: Sampling temperature (0.0 = greedy).
        top_k: Top-K filtering parameter.
        top_p: Top-P nucleus sampling parameter.
        eos_token_id: End of sequence token ID to terminate generation.
        device: Torch execution device.
        max_seq_len: Pre-allocated maximum sequence capacity for the cache.
        time_budget_s: Optional time budget in seconds. If exceeded, stops early.

    Returns:
        GenerationOutput containing generated token IDs, decoded text, and timing metrics.
    """
    model.eval()

    if isinstance(prompts, str):
        prompts = [prompts]

    # Tokenize prompts
    encoded = tokenizer(prompts, return_tensors="pt")
    input_ids = encoded["input_ids"].to(device)
    batch_size, prompt_tokens = input_ids.shape

    if eos_token_id is None:
        eos_token_id = getattr(tokenizer, "eos_token_id", None)

    # Allocate KV Cache
    cache_capacity = max(max_seq_len, prompt_tokens + max_new_tokens + 16)
    cache = KVCache.from_config(
        model.config,
        batch_size=batch_size,
        max_seq_len=cache_capacity,
        device=torch.device(device),
        dtype=next(model.parameters()).dtype,
    )

    curr_input_ids = input_ids.clone()
    generated_token_list: List[torch.Tensor] = []

    # Benchmark timers
    t_start = time.perf_counter()
    t_first_token: Optional[float] = None

    with torch.no_grad():
        # --- Prefill Phase ---
        # Forward all prompt tokens at start_pos = 0
        prefill_logits = model(input_ids, kv_cache=cache, start_pos=0)
        # Logits at the last prompt position: (batch_size, vocab_size)
        next_token_logits = prefill_logits[:, -1, :]
        next_token = sample(
            next_token_logits,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
        )
        t_first_token = time.perf_counter()

        generated_token_list.append(next_token)
        curr_input_ids = torch.cat([curr_input_ids, next_token], dim=-1)

        # Check early exit if prompt + 1 generated token hits EOS
        should_stop = eos_token_id is not None and (next_token == eos_token_id).all()

        # --- Decode Phase ---
        # Feed 1 token at a time at start_pos = prompt_tokens + step
        curr_token = next_token
        start_pos = prompt_tokens

        for step in range(1, max_new_tokens):
            if should_stop:
                break
            if time_budget_s is not None and (time.perf_counter() - t_start) > time_budget_s:
                break

            # Forward only the single newest token across all 9 heads
            decode_logits = model(curr_token, kv_cache=cache, start_pos=start_pos)
            next_token_logits = decode_logits[:, -1, :]

            next_token = sample(
                next_token_logits,
                temperature=temperature,
                top_k=top_k,
                top_p=top_p,
            )

            generated_token_list.append(next_token)
            curr_input_ids = torch.cat([curr_input_ids, next_token], dim=-1)
            curr_token = next_token
            start_pos += 1

            if eos_token_id is not None and (next_token == eos_token_id).all():
                break

    t_end = time.perf_counter()

    ttft_ms = (t_first_token - t_start) * 1000.0 if t_first_token is not None else 0.0
    total_time_ms = (t_end - t_start) * 1000.0

    generated_tokens_count = len(generated_token_list)
    decode_tokens_count = max(0, generated_tokens_count - 1)
    decode_time_s = (t_end - t_first_token) if t_first_token is not None else 0.0

    tokens_per_second = (
        (decode_tokens_count / decode_time_s) if decode_time_s > 0 else 0.0
    )
    ms_per_token = (
        (decode_time_s * 1000.0 / decode_tokens_count) if decode_tokens_count > 0 else 0.0
    )

    # Decode text
    decoded_texts = [
        tokenizer.decode(curr_input_ids[i], skip_special_tokens=True)
        for i in range(curr_input_ids.shape[0])
    ]

    return GenerationOutput(
        token_ids=curr_input_ids,
        text=decoded_texts,
        ttft_ms=ttft_ms,
        total_time_ms=total_time_ms,
        prompt_tokens=prompt_tokens,
        generated_tokens=generated_tokens_count,
        tokens_per_second=tokens_per_second,
        ms_per_token=ms_per_token,
    )
