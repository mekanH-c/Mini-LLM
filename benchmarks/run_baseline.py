"""Benchmark script for Phase 0-1 Naive Inference Baseline vs Hugging Face."""

import os
import sys
import time
import psutil
import torch
from typing import List, Dict, Any
from transformers import AutoModelForCausalLM, AutoTokenizer

# Add src to sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src")))

from mini_infer.loader import load_from_hf
from mini_infer.generate import generate

MODEL_ID = "HuggingFaceTB/SmolLM-135M"

BENCHMARK_PROMPTS = [
    "The capital of France is",
    "def fibonacci(n):",
    "The quick brown fox jumps",
    "In mathematics, the Riemann hypothesis is",
    "Machine learning models require data to",
]


def get_peak_memory_mb() -> float:
    """Returns current process Resident Set Size (RSS) in MB."""
    process = psutil.Process()
    return process.memory_info().rss / (1024 * 1024)


def benchmark_custom(
    model: Any,
    tokenizer: Any,
    prompts: List[str],
    max_new_tokens: int = 32,
    num_runs: int = 3,
) -> Dict[str, float]:
    """Benchmark custom naive inference loop."""
    # Warmup
    _ = generate(model, tokenizer, prompts[0], max_new_tokens=4, temperature=0.0)

    ttft_list = []
    tps_list = []
    total_time_list = []

    for run_idx in range(num_runs):
        for prompt in prompts:
            out = generate(
                model,
                tokenizer,
                prompt,
                max_new_tokens=max_new_tokens,
                temperature=0.0,
            )
            ttft_list.append(out.ttft_ms)
            tps_list.append(out.tokens_per_second)
            total_time_list.append(out.total_time_ms)

    return {
        "ttft_ms": sum(ttft_list) / len(ttft_list),
        "tokens_per_sec": sum(tps_list) / len(tps_list),
        "total_time_ms": sum(total_time_list) / len(total_time_list),
        "peak_rss_mb": get_peak_memory_mb(),
    }


def benchmark_hf(
    hf_model: Any,
    tokenizer: Any,
    prompts: List[str],
    max_new_tokens: int = 32,
    num_runs: int = 3,
) -> Dict[str, float]:
    """Benchmark Hugging Face generate() on identical prompts."""
    # Warmup
    dummy_in = tokenizer(prompts[0], return_tensors="pt")["input_ids"]
    with torch.no_grad():
        _ = hf_model.generate(dummy_in, max_new_tokens=4, do_sample=False)

    ttft_list = []
    tps_list = []
    total_time_list = []

    for run_idx in range(num_runs):
        for prompt in prompts:
            input_ids = tokenizer(prompt, return_tensors="pt")["input_ids"]
            prompt_len = input_ids.shape[1]

            t_start = time.perf_counter()
            with torch.no_grad():
                # Step 1: measure first token time (TTFT)
                first_out = hf_model(input_ids)
                _ = first_out.logits[:, -1, :].argmax(dim=-1)
                t_first = time.perf_counter()

                # Full generate
                full_out = hf_model.generate(
                    input_ids,
                    max_new_tokens=max_new_tokens,
                    do_sample=False,
                )
                t_end = time.perf_counter()

            ttft_ms = (t_first - t_start) * 1000.0
            total_time_ms = (t_end - t_start) * 1000.0
            generated_tokens = full_out.shape[1] - prompt_len
            decode_time_s = t_end - t_first
            decode_tokens = max(0, generated_tokens - 1)
            tps = (decode_tokens / decode_time_s) if decode_time_s > 0 else 0.0

            ttft_list.append(ttft_ms)
            tps_list.append(tps)
            total_time_list.append(total_time_ms)

    return {
        "ttft_ms": sum(ttft_list) / len(ttft_list),
        "tokens_per_sec": sum(tps_list) / len(tps_list),
        "total_time_ms": sum(total_time_list) / len(total_time_list),
        "peak_rss_mb": get_peak_memory_mb(),
    }


def main():
    print(f"Loading {MODEL_ID} for benchmark comparison...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)

    print("Loading Custom Engine in float32...")
    custom_model, _ = load_from_hf(MODEL_ID, dtype=torch.float32)
    custom_model.eval()

    print("Loading Hugging Face model in float32...")
    hf_model = AutoModelForCausalLM.from_pretrained(MODEL_ID, torch_dtype=torch.float32)
    hf_model.eval()

    max_new_tokens = 32
    num_runs = 3
    print(f"\nRunning benchmark across {len(BENCHMARK_PROMPTS)} prompts ({num_runs} iterations, {max_new_tokens} tokens generated per prompt)...")

    print("\n--- Running Custom Naive Inference Engine ---")
    custom_metrics = benchmark_custom(
        custom_model, tokenizer, BENCHMARK_PROMPTS, max_new_tokens, num_runs
    )
    print(f"Custom Naive Engine Results:")
    print(f"  TTFT:               {custom_metrics['ttft_ms']:.2f} ms")
    print(f"  Decode Throughput:  {custom_metrics['tokens_per_sec']:.2f} tokens/s")
    print(f"  Total Latency:      {custom_metrics['total_time_ms']:.2f} ms")
    print(f"  Peak RSS Memory:    {custom_metrics['peak_rss_mb']:.2f} MB")

    print("\n--- Running Hugging Face Reference (hf_model.generate) ---")
    hf_metrics = benchmark_hf(
        hf_model, tokenizer, BENCHMARK_PROMPTS, max_new_tokens, num_runs
    )
    print(f"Hugging Face Generate Results:")
    print(f"  TTFT:               {hf_metrics['ttft_ms']:.2f} ms")
    print(f"  Decode Throughput:  {hf_metrics['tokens_per_sec']:.2f} tokens/s")
    print(f"  Total Latency:      {hf_metrics['total_time_ms']:.2f} ms")
    print(f"  Peak RSS Memory:    {hf_metrics['peak_rss_mb']:.2f} MB")

    print("\n=== SUMMARY TABLE (CPU Baseline: Phase 0-1) ===")
    print(f"{'Engine':<25} | {'TTFT (ms)':<12} | {'Tokens / sec':<14} | {'Total Time (ms)':<16} | {'Peak Memory (MB)':<16}")
    print("-" * 92)
    print(f"{'Custom Naive Engine':<25} | {custom_metrics['ttft_ms']:<12.2f} | {custom_metrics['tokens_per_sec']:<14.2f} | {custom_metrics['total_time_ms']:<16.2f} | {custom_metrics['peak_rss_mb']:<16.2f}")
    print(f"{'Hugging Face (generate)':<25} | {hf_metrics['ttft_ms']:<12.2f} | {hf_metrics['tokens_per_sec']:<14.2f} | {hf_metrics['total_time_ms']:<16.2f} | {hf_metrics['peak_rss_mb']:<16.2f}")


if __name__ == "__main__":
    main()
