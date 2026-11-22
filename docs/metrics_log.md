# Metrics Log

This log tracks measured performance metrics across each phase of development. 

> **Important**: No metrics are fabricated. Every entry corresponds to an actual benchmark run, recording hardware details, library versions, model, and configuration parameters.

| Date | Phase / Change | Hardware / Device | Model | Prompt Len / Gen Len | Batch Size | Tokens / sec | TTFT (ms) | Peak Memory | Notes |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| 2026-11-21 | Phase 0-1 Custom Naive Baseline | Intel x64 CPU (12 cores) | SmolLM-135M (float32) | avg ~6 / 32 | 1 | 8.42 | 92.26 | 1427.4 MB RSS | Full-sequence recomputation per token (no KV cache). PyTorch 2.15.0+cpu. |
| 2026-11-21 | Phase 0-1 Hugging Face Reference | Intel x64 CPU (12 cores) | SmolLM-135M (float32) | avg ~6 / 32 | 1 | 15.64 | 77.51 | 1423.2 MB RSS | HF `generate()` on same 5 prompts (uses internal KV cache). Transformers 5.23.0. |
| 2026-11-22 | Phase 0-1 Custom Naive Baseline | NVIDIA Tesla T4 (Kaggle) | SmolLM-135M (float32) | avg ~6 / 32 | 1 | 29.88 | 32.63 | 1097.13 MB VRAM | Custom naive engine. At 32 new tokens the small 135M model is kernel launch overhead-bound, so KV cache benefit is small here. Both models were resident on GPU during initial run (~540MB each). PyTorch 2.5+, Transformers 4.47+. |
| 2026-11-22 | Phase 0-1 Hugging Face Reference | NVIDIA Tesla T4 (Kaggle) | SmolLM-135M (float32) | avg ~6 / 32 | 1 | 28.80 | 39.06 | 1086.84 MB VRAM | HF `generate()` on identical 5 prompts (3 runs each, 15 iterations averaged). Kernel overhead-bound at short 32-token generation length. PyTorch 2.5+, Transformers 4.47+. |
