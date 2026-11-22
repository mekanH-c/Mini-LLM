"""KV Cache Benchmark: Naive vs KV-Cached vs Hugging Face across sequence lengths."""

import argparse
import gc
import os
import sys
import time
import psutil
import torch
from typing import Dict, List, Optional
import matplotlib.pyplot as plt
from transformers import AutoModelForCausalLM, AutoTokenizer

# Add src to sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src")))

from mini_infer.loader import load_from_hf
from mini_infer.generate import generate, generate_cached

MODEL_ID = "HuggingFaceTB/SmolLM-135M"
TEST_PROMPT = "The primary purpose of an operating system is"


def get_peak_memory_mb(device: str = "cpu") -> float:
    """Returns peak memory: CUDA peak VRAM if cuda, else process RSS in MB."""
    if "cuda" in device and torch.cuda.is_available():
        return torch.cuda.max_memory_allocated() / (1024 * 1024)
    process = psutil.Process()
    return process.memory_info().rss / (1024 * 1024)


def reset_memory(device: str = "cpu"):
    """Resets memory tracking and cleans up cache."""
    gc.collect()
    if "cuda" in device and torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()


def benchmark_custom_cached_runs(
    model, tokenizer, prompt: str, gen_len: int, num_runs: int = 3, device: str = "cpu"
) -> Dict[str, float]:
    """Benchmark Custom Cached engine, averaging across multiple runs."""
    ttft_list = []
    tps_list = []
    ms_per_tok_list = []
    total_time_list = []

    for _ in range(num_runs):
        reset_memory(device)
        out = generate_cached(
            model=model,
            tokenizer=tokenizer,
            prompts=prompt,
            max_new_tokens=gen_len,
            temperature=0.0,
            device=device,
        )
        ttft_list.append(out.ttft_ms)
        tps_list.append(out.tokens_per_second)
        ms_per_tok_list.append(out.ms_per_token)
        total_time_list.append(out.total_time_ms)

    return {
        "ttft_ms": sum(ttft_list) / len(ttft_list),
        "tokens_per_sec": sum(tps_list) / len(tps_list),
        "ms_per_token": sum(ms_per_tok_list) / len(ms_per_tok_list),
        "total_time_s": (sum(total_time_list) / len(total_time_list)) / 1000.0,
        "peak_mem_mb": get_peak_memory_mb(device),
        "runs": num_runs,
    }


def benchmark_hf_cached_runs(
    hf_model, tokenizer, prompt: str, gen_len: int, num_runs: int = 3, device: str = "cpu"
) -> Dict[str, float]:
    """Benchmark Hugging Face generate() (Python/PyTorch with internal KV cache)."""
    input_ids = tokenizer(prompt, return_tensors="pt")["input_ids"].to(device)
    prompt_len = input_ids.shape[1]

    ttft_list = []
    tps_list = []
    ms_per_tok_list = []
    total_time_list = []

    for _ in range(num_runs):
        reset_memory(device)
        t_start = time.perf_counter()
        with torch.no_grad():
            # Measure TTFT (prefill forward pass)
            first_out = hf_model(input_ids)
            _ = first_out.logits[:, -1, :].argmax(dim=-1)
            t_first = time.perf_counter()

            # Full generate
            full_out = hf_model.generate(input_ids, max_new_tokens=gen_len, do_sample=False)
            t_end = time.perf_counter()

        ttft_ms = (t_first - t_start) * 1000.0
        total_time_ms = (t_end - t_start) * 1000.0
        generated_tokens = full_out.shape[1] - prompt_len
        decode_tokens = max(0, generated_tokens - 1)
        decode_time_s = t_end - t_first
        tps = (decode_tokens / decode_time_s) if decode_time_s > 0 else 0.0
        ms_per_tok = (decode_time_s * 1000.0 / decode_tokens) if decode_tokens > 0 else 0.0

        ttft_list.append(ttft_ms)
        tps_list.append(tps)
        ms_per_tok_list.append(ms_per_tok)
        total_time_list.append(total_time_ms)

    return {
        "ttft_ms": sum(ttft_list) / len(ttft_list),
        "tokens_per_sec": sum(tps_list) / len(tps_list),
        "ms_per_token": sum(ms_per_tok_list) / len(ms_per_tok_list),
        "total_time_s": (sum(total_time_list) / len(total_time_list)) / 1000.0,
        "peak_mem_mb": get_peak_memory_mb(device),
        "runs": num_runs,
    }


def benchmark_custom_naive_runs(
    model,
    tokenizer,
    prompt: str,
    gen_len: int,
    num_runs: int = 1,
    time_budget_s: Optional[float] = None,
    device: str = "cpu",
) -> Dict[str, float]:
    """Benchmark Naive engine (full recomputation per token)."""
    ttft_list = []
    tps_list = []
    ms_per_tok_list = []
    total_time_list = []
    tokens_gen_list = []

    for _ in range(num_runs):
        reset_memory(device)
        out = generate(
            model=model,
            tokenizer=tokenizer,
            prompts=prompt,
            max_new_tokens=gen_len,
            temperature=0.0,
            device=device,
            time_budget_s=time_budget_s,
        )
        ttft_list.append(out.ttft_ms)
        tps_list.append(out.tokens_per_second)
        ms_per_tok_list.append(out.ms_per_token)
        total_time_list.append(out.total_time_ms)
        tokens_gen_list.append(out.generated_tokens)

    return {
        "ttft_ms": sum(ttft_list) / len(ttft_list),
        "tokens_per_sec": sum(tps_list) / len(tps_list),
        "ms_per_token": sum(ms_per_tok_list) / len(ms_per_tok_list),
        "total_time_s": (sum(total_time_list) / len(total_time_list)) / 1000.0,
        "peak_mem_mb": get_peak_memory_mb(device),
        "tokens_generated": sum(tokens_gen_list) / len(tokens_gen_list),
        "runs": num_runs,
    }


def save_scaling_plot(results: List[Dict], output_path: str = "benchmarks/kv_cache_scaling.png"):
    """Generates and saves a scaling comparison chart (Speed vs Length)."""
    lengths = [r["length"] for r in results]
    cached_tps = [r["cached"]["tokens_per_sec"] for r in results]
    hf_tps = [r["hf"]["tokens_per_sec"] for r in results]

    naive_lengths = [r["length"] for r in results if r["naive"] is not None]
    naive_tps = [r["naive"]["tokens_per_sec"] for r in results if r["naive"] is not None]

    plt.figure(figsize=(9, 5.5), dpi=150)
    plt.plot(lengths, cached_tps, marker="o", linewidth=2.5, color="#1f77b4", label="Custom KV Cache (O(N))")
    plt.plot(lengths, hf_tps, marker="s", linewidth=2.0, linestyle="--", color="#2ca02c", label="Hugging Face generate() [Cached]")
    if naive_lengths:
        plt.plot(naive_lengths, naive_tps, marker="^", linewidth=2.5, color="#d62728", label="Custom Naive (O(N^2) Recomputation)")

    plt.title("Autoregressive Generation Throughput vs Sequence Length\n(SmolLM-135M, float32)", fontsize=13, fontweight="bold")
    plt.xlabel("Generated Sequence Length (Tokens)", fontsize=11)
    plt.ylabel("Decode Throughput (Tokens / Second)", fontsize=11)
    plt.grid(True, linestyle=":", alpha=0.6)
    plt.legend(fontsize=10, loc="best")
    plt.xticks(lengths)

    # Annotate speedup at longest common length
    if naive_lengths:
        max_naive_l = max(naive_lengths)
        idx = lengths.index(max_naive_l)
        speedup = cached_tps[idx] / naive_tps[-1]
        plt.annotate(
            f"{speedup:.1f}x Speedup",
            xy=(max_naive_l, cached_tps[idx]),
            xytext=(max_naive_l, cached_tps[idx] + 1.5),
            ha="center",
            fontweight="bold",
            color="#1f77b4",
            arrowprops=dict(arrowstyle="->", color="#1f77b4", lw=1.5),
        )

    plt.tight_layout()
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    plt.savefig(output_path)
    plt.close()
    print(f"\n[Chart Saved] Throughput scaling plot saved to: {output_path}")


def run_benchmarks(
    lengths: List[int],
    device: str = "cpu",
    cached_runs: int = 3,
    hf_runs: int = 3,
    naive_128_runs: int = 3,
    naive_time_budget_s: float = 600.0,
    plot_path: str = "benchmarks/kv_cache_scaling.png",
):
    print("=" * 95)
    print(f"Mini-LLM KV Cache Scaling Benchmark (SmolLM-135M, float32, device={device})")
    print(f"Prompt: '{TEST_PROMPT}'")
    print(f"Lengths: {lengths}")
    print(f"Cached & HF runs per length: {cached_runs} | Naive time budget: {naive_time_budget_s:.0f}s")
    print("=" * 95)

    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)

    # Model profiling with isolated memory
    print("\n[1/3] Benchmarking Custom Cached Engine...")
    custom_model, _ = load_from_hf(MODEL_ID, dtype=torch.float32)
    custom_model.to(device)
    custom_model.eval()

    # Warmup
    _ = generate_cached(custom_model, tokenizer, TEST_PROMPT, max_new_tokens=4, device=device)

    cached_results = {}
    for length in lengths:
        print(f"  Testing Cached length {length} ({cached_runs} runs)...", end="", flush=True)
        res = benchmark_custom_cached_runs(custom_model, tokenizer, TEST_PROMPT, length, num_runs=cached_runs, device=device)
        cached_results[length] = res
        print(f" Done ({res['tokens_per_sec']:.2f} tok/s, TTFT: {res['ttft_ms']:.1f}ms, decode: {res['ms_per_token']:.1f}ms/tok)")

    # Delete custom model to measure HF memory in complete isolation
    del custom_model
    reset_memory(device)

    print("\n[2/3] Benchmarking Hugging Face Reference (Python/PyTorch with KV Cache)...")
    hf_model = AutoModelForCausalLM.from_pretrained(MODEL_ID, torch_dtype=torch.float32)
    hf_model.to(device)
    hf_model.eval()

    # Warmup
    hf_in = tokenizer(TEST_PROMPT, return_tensors="pt")["input_ids"].to(device)
    with torch.no_grad():
        _ = hf_model.generate(hf_in, max_new_tokens=4, do_sample=False)

    hf_results = {}
    for length in lengths:
        print(f"  Testing HF length {length} ({hf_runs} runs)...", end="", flush=True)
        res = benchmark_hf_cached_runs(hf_model, tokenizer, TEST_PROMPT, length, num_runs=hf_runs, device=device)
        hf_results[length] = res
        print(f" Done ({res['tokens_per_sec']:.2f} tok/s, TTFT: {res['ttft_ms']:.1f}ms, decode: {res['ms_per_token']:.1f}ms/tok)")

    del hf_model
    reset_memory(device)

    print("\n[3/3] Benchmarking Custom Naive Engine...")
    custom_model_naive, _ = load_from_hf(MODEL_ID, dtype=torch.float32)
    custom_model_naive.to(device)
    custom_model_naive.eval()

    naive_results = {}
    for length in lengths:
        # Determine runs for naive: 3 runs for <= 128, 1 run for 512, 1 run with budget for 2048
        if length <= 128:
            n_runs = naive_128_runs
        else:
            n_runs = 1

        print(f"  Testing Naive length {length} ({n_runs} run(s), budget={naive_time_budget_s:.0f}s)...", end="", flush=True)
        res = benchmark_custom_naive_runs(
            custom_model_naive,
            tokenizer,
            TEST_PROMPT,
            gen_len=length,
            num_runs=n_runs,
            time_budget_s=naive_time_budget_s,
            device=device,
        )
        naive_results[length] = res
        print(f" Done ({res['tokens_per_sec']:.2f} tok/s, TTFT: {res['ttft_ms']:.1f}ms, decode: {res['ms_per_token']:.1f}ms/tok)")

    del custom_model_naive
    reset_memory(device)

    # Format structured results
    combined_results = []
    for length in lengths:
        combined_results.append({
            "length": length,
            "cached": cached_results[length],
            "hf": hf_results[length],
            "naive": naive_results.get(length),
        })

    # Print Detailed Table
    print("\n" + "=" * 120)
    print("PHASE 2 DETAILED BENCHMARK RESULTS")
    print("=" * 120)
    header = (
        f"{'Length':<8} | {'Engine':<20} | {'Throughput':<14} | {'TTFT (Prefill)':<16} | "
        f"{'Decode Time':<16} | {'Total Time':<12} | {'Runs':<6}"
    )
    print(header)
    print("-" * 120)

    for r in combined_results:
        L = r["length"]
        c = r["cached"]
        h = r["hf"]
        n = r["naive"]

        print(f"{L:<8} | {'Custom Cached':<20} | {c['tokens_per_sec']:<8.2f} tok/s | {c['ttft_ms']:<10.2f} ms   | {c['ms_per_token']:<10.2f} ms/tok | {c['total_time_s']:<8.2f} s   | {c['runs']:<6}")
        print(f"{'':<8} | {'Hugging Face':<20} | {h['tokens_per_sec']:<8.2f} tok/s | {h['ttft_ms']:<10.2f} ms   | {h['ms_per_token']:<10.2f} ms/tok | {h['total_time_s']:<8.2f} s   | {h['runs']:<6}")
        if n is not None:
            speedup = c['tokens_per_sec'] / n['tokens_per_sec'] if n['tokens_per_sec'] > 0 else 0.0
            print(f"{'':<8} | {'Custom Naive':<20} | {n['tokens_per_sec']:<8.2f} tok/s | {n['ttft_ms']:<10.2f} ms   | {n['ms_per_token']:<10.2f} ms/tok | {n['total_time_s']:<8.2f} s   | {n['runs']:<6} (Speedup: {speedup:.2f}x)")
        print("-" * 120)

    # Save scaling chart
    save_scaling_plot(combined_results, plot_path)

    return combined_results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="KV Cache Benchmark Harness")
    parser.add_argument(
        "--lengths",
        nargs="+",
        type=int,
        default=[128, 512],
        help="List of token generation lengths to benchmark (default: 128 512)",
    )
    parser.add_argument("--device", type=str, default="cpu", help="Device (cpu or cuda)")
    parser.add_argument("--cached-runs", type=int, default=3, help="Number of runs for cached engine (default: 3)")
    parser.add_argument("--hf-runs", type=int, default=3, help="Number of runs for Hugging Face (default: 3)")
    parser.add_argument("--naive-128-runs", type=int, default=3, help="Number of runs for naive at 128 (default: 3)")
    parser.add_argument("--time-budget", type=float, default=600.0, help="Time budget in seconds for naive (default: 600)")
    parser.add_argument("--plot-path", type=str, default="benchmarks/kv_cache_scaling.png", help="Path to save chart PNG")
    args = parser.parse_args()

    run_benchmarks(
        lengths=args.lengths,
        device=args.device,
        cached_runs=args.cached_runs,
        hf_runs=args.hf_runs,
        naive_128_runs=args.naive_128_runs,
        naive_time_budget_s=args.time_budget,
        plot_path=args.plot_path,
    )
