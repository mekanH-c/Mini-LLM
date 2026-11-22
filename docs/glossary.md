# LLM Systems & Inference Engine Glossary

Key concepts, architectural terms, and performance metrics explained simply for interviews.

### Core Architecture
- **RMSNorm (Root Mean Square Normalization)**: A simplification of LayerNorm that scales activations by the root mean square of inputs without subtracting the mean, reducing computational overhead while maintaining stability.
- **RoPE (Rotary Position Embedding)**: Encodes positional information by multiplying query and key vectors by a 2D rotation matrix in the complex plane. RoPE preserves relative distances and generalizes well to sequence lengths.
- **MHA / MQA / GQA (Multi-Head / Multi-Query / Grouped-Query Attention)**:
  - *MHA*: Each attention query head has its own key and value head ($H_Q = H_{KV}$). High memory footprint for KV cache.
  - *MQA*: All query heads share a single key and value head ($H_{KV} = 1$). Fast, low memory, but potential degradation in quality.
  - *GQA*: Query heads are partitioned into groups, where each group shares a key and value head ($1 < H_{KV} < H_Q$). SmolLM-135M uses GQA (e.g., 9 query heads, 3 KV heads).
- **SwiGLU (Swish Gated Linear Unit)**: A non-linear feedforward layer that computes $\text{SwiGLU}(x) = (\text{Linear}_1(x) \odot \text{SiLU}(\text{Linear}_2(x))) \times W_{down}$. Provides higher expressivity than standard ReLU/GELU FFNs.
- **Tied Word Embeddings**: Architectural technique where the language model head projection weights (`lm_head.weight`) are shared directly with the input token embedding table (`model.embed_tokens.weight`), saving millions of parameters in memory without degrading token representation.

### Sampling Strategies
- **Greedy Decoding**: Deterministic token selection choosing $\text{argmax}(\text{logits})$. Zero stochasticity.
- **Temperature Scaling**: Dividing logits by scalar $T > 0$ before softmax ($z_i / T$). High $T$ flattens the distribution towards uniform randomness; low $T$ sharpens it toward greedy argmax.
- **Top-$K$ Filtering**: Retains only the $K$ tokens with the highest unnormalized logits, masking all other tokens to $-\infty$.
- **Top-$P$ (Nucleus) Sampling**: Sorts tokens descending by probability and computes the cumulative distribution function (CDF). Retains the smallest set of tokens whose cumulative mass exceeds threshold $P$, dynamically adapting the candidate pool size based on distribution certainty.

### Numerical Formats & Precision
- **FP32 (Single Precision)**: 1 sign bit, 8 exponent bits, 23 mantissa bits. Full precision standard.
- **BF16 (Brain Floating Point)**: 1 sign bit, 8 exponent bits (same dynamic range as FP32), 7 mantissa bits. Reduces memory by $2\times$ while avoiding FP16 underflow/overflow scaling issues, but has lower precision per operation.

### Inference & Serving
- **Prefill (Prompt Processing)**: The compute-bound phase where the entire prompt sequence is processed in parallel to generate initial KV caches and the first output token.
- **Decode (Token Generation)**: The memory-bandwidth-bound phase where tokens are generated autoregressively one by one, reading the KV cache for all previous tokens at each step.
- **KV Cache (Key-Value Cache)**: Stores key and value projections across all transformer layers for historical tokens ($0 \dots t$). Decouples autoregressive decoding so each step only computes projections for the single newest token, reducing generation time complexity from $O(N^2)$ to $O(N)$.
- **KV Cache Memory Footprint Formula**: The memory occupied by past keys and values across the model:
  $$\text{Memory (bytes)} = 2 \times L \times B \times H_{KV} \times S \times d_{\text{head}} \times \text{bytes\_per\_elem}$$
  For SmolLM-135M ($L=30, B=1, H_{KV}=3, S=2048, d_{\text{head}}=64$, float32), the full cache occupies exactly $94,371,840 \text{ bytes} \approx 94.37 \text{ MB}$ ($90 \text{ MiB}$).
- **Decode Phase Rotary Alignment**: In decode, the single query token across all heads at position $t$ is rotated using angles derived from $t$, guaranteeing that dot products $\langle R_t q, R_s k \rangle = q^T R_{t-s} k$ yield identical relative position attention scores as a full-sequence forward pass.
- **TTFT (Time to First Token)**: Time taken from request arrival until the first token is emitted (dominated by prefill latency).
- **TPOT (Time Per Output Token) / Inter-token Latency**: Average time to generate each subsequent token during the decode phase.
- **Continuous Batching (Iteration-level Scheduling)**: Serving paradigm where requests enter and exit the batch on each token generation iteration rather than waiting for an entire batch to finish.
- **Paged Attention**: Memory management scheme inspired by virtual memory OS paging. KV cache is divided into non-contiguous physical blocks managed via page tables, eliminating internal and external fragmentation.
