"""Naive autoregressive text generation loop with performance metrics."""

import time
from dataclasses import dataclass
from typing import Any, List, Optional, Union
import torch

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
    )
