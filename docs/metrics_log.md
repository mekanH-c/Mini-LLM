# Metrics Log

This log tracks measured performance metrics across each phase of development. 

> **Important**: No metrics are fabricated. Every entry corresponds to an actual benchmark run, recording hardware details, library versions, model, and configuration parameters.

| Date | Phase / Change | Hardware / Device | Model | Prompt Len / Gen Len | Batch Size | Tokens / sec | TTFT (ms) | Peak Memory | Notes |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| 2026-11-21 | Phase 0-1 Custom Naive Baseline | Intel x64 CPU (12 cores) | SmolLM-135M (float32) | avg ~6 / 32 | 1 | 8.42 | 92.26 | 1427.4 MB RSS | Full-sequence recomputation per token (no KV cache). PyTorch 2.15.0+cpu. |
| 2026-11-21 | Phase 0-1 Hugging Face Reference | Intel x64 CPU (12 cores) | SmolLM-135M (float32) | avg ~6 / 32 | 1 | 15.64 | 77.51 | 1423.2 MB RSS | HF `generate()` on same 5 prompts (uses internal KV cache). Transformers 5.23.0. |
| 2026-11-22 | Phase 0-1 Custom Naive Baseline | Tesla T4 (Kaggle) | SmolLM-135M (float32) | avg ~6 / 32 | 1 | 27.58 | 35.08 | 573.67 MB VRAM | No KV cache. 15 runs (5 prompts x 3). PyTorch 2.11.0+cu130, Transformers 5.20.2, Safetensors 0.8.0. An earlier run gave 29.88 tokens/s, so run-to-run variation is about 8%. At 32 tokens the small model is kernel-launch-bound, so the KV cache gives little gain here. |
| 2026-11-22 | Phase 0-1 Hugging Face Reference | Tesla T4 (Kaggle) | SmolLM-135M (float32) | avg ~6 / 32 | 1 | 26.53 | 42.36 | 533.37 MB VRAM | HF generate() on the same prompts, same 15 runs. Same library versions. Engines are tied at this length. Earlier 1097/1087 MB memory figures were wrong because both models were loaded at once. |
| 2026-11-22 | Phase 2 Custom KV Cache (128) | Intel x64 CPU (12 cores) | SmolLM-135M (float32) | ~8 / 128 | 1 | 13.57 | 102.33 | 1431.2 MB RSS | Custom prefill + decode with KVCache. Target length is total sequence length (capped at 2048). 3 runs averaged. Decode: 73.73 ms/token. Total time: 9.47 s. PyTorch 2.15.0+cpu. |
| 2026-11-22 | Phase 2 Hugging Face Reference (128) | Intel x64 CPU (12 cores) | SmolLM-135M (float32) | ~8 / 128 | 1 | 10.46 | 130.65 | 1428.5 MB RSS | HF generate() (Python/PyTorch with internal KV cache). 3 runs averaged. Decode: 95.62 ms/token. Total time: 12.27 s. |
| 2026-11-22 | Phase 2 Custom Naive Baseline (128) | Intel x64 CPU (12 cores) | SmolLM-135M (float32) | ~8 / 128 | 1 | 2.70 | 145.40 | 1435.0 MB RSS | Naive recomputation per token. 3 runs averaged. Decode: 370.83 ms/token. Total time: 47.24 s. Custom cached speedup: 5.03x. |
| 2026-11-22 | Phase 2 Custom KV Cache (512) | Intel x64 CPU (12 cores) | SmolLM-135M (float32) | ~8 / 512 | 1 | 9.27 | 137.54 | 1445.6 MB RSS | Custom prefill + decode with KVCache. Target length is total sequence length (capped at 2048). 3 runs averaged. Decode: 107.83 ms/token. Total time: 55.24 s. PyTorch 2.15.0+cpu. |
| 2026-11-22 | Phase 2 Hugging Face Reference (512) | Intel x64 CPU (12 cores) | SmolLM-135M (float32) | ~8 / 512 | 1 | 10.11 | 127.89 | 1439.1 MB RSS | HF generate() (Python/PyTorch with internal KV cache). 3 runs averaged. Decode: 98.96 ms/token. Total time: 50.70 s. |
| 2026-11-22 | Phase 2 Custom Naive Baseline (512) | Intel x64 CPU (12 cores) | SmolLM-135M (float32) | ~8 / 512 | 1 | 0.94 | 143.25 | 1448.2 MB RSS | Naive recomputation per token. 1 run due to quadratic decode time (~9.04 min). Decode: 1060.74 ms/token. Total time: 542.18 s. Custom cached speedup: 9.84x. |
