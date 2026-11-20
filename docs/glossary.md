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

### Inference & Serving
- **Prefill (Prompt Processing)**: The compute-bound phase where the entire prompt sequence is processed in parallel to generate initial KV caches and the first output token.
- **Decode (Token Generation)**: The memory-bandwidth-bound phase where tokens are generated autoregressively one by one, reading the KV cache for all previous tokens at each step.
- **TTFT (Time to First Token)**: Time taken from request arrival until the first token is emitted (dominated by prefill latency).
- **TPOT (Time Per Output Token) / Inter-token Latency**: Average time to generate each subsequent token during the decode phase.
- **Continuous Batching (Iteration-level Scheduling)**: Serving paradigm where requests enter and exit the batch on each token generation iteration rather than waiting for an entire batch to finish.
- **Paged Attention**: Memory management scheme inspired by virtual memory OS paging. KV cache is divided into non-contiguous physical blocks managed via page tables, eliminating internal and external fragmentation.
