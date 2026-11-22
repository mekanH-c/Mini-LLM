# Metrics Log

This log tracks measured performance metrics across each phase of development. 

> **Important**: No metrics are fabricated. Every entry corresponds to an actual benchmark run, recording hardware details, library versions, model, and configuration parameters.

| Date | Phase / Change | Hardware / Device | Model | Prompt Len / Gen Len | Batch Size | Tokens / sec | TTFT (ms) | Peak Memory | Notes |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| 2026-11-21 | Phase 0-1 Custom Naive Baseline | Intel x64 CPU (12 cores) | SmolLM-135M (float32) | avg ~6 / 32 | 1 | 8.42 | 92.26 | 1427.4 MB RSS | Full-sequence recomputation per token (no KV cache). PyTorch 2.15.0+cpu. |
| 2026-11-21 | Phase 0-1 Hugging Face Reference | Intel x64 CPU (12 cores) | SmolLM-135M (float32) | avg ~6 / 32 | 1 | 15.64 | 77.51 | 1423.2 MB RSS | HF `generate()` on same 5 prompts (uses internal KV cache). Transformers 5.23.0. |
| 2026-11-22 | Phase 0-1 Custom Naive Baseline | Tesla T4 (Kaggle) | SmolLM-135M (float32) | avg ~6 / 32 | 1 | 27.58 | 35.08 | 573.67 MB VRAM | No KV cache. 15 runs (5 prompts x 3). PyTorch 2.11.0+cu130, Transformers 5.20.2, Safetensors 0.8.0. An earlier run gave 29.88 tokens/s, so run-to-run variation is about 8%. At 32 tokens the small model is kernel-launch-bound, so the KV cache gives little gain here. |
| 2026-11-22 | Phase 0-1 Hugging Face Reference | Tesla T4 (Kaggle) | SmolLM-135M (float32) | avg ~6 / 32 | 1 | 26.53 | 42.36 | 533.37 MB VRAM | HF generate() on the same prompts, same 15 runs. Same library versions. Engines are tied at this length. Earlier 1097/1087 MB memory figures were wrong because both models were loaded at once. |
