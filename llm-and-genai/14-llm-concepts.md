# Chapter 14: LLM Concepts

This chapter covers the fundamentals of Large Language Models — transformer architecture, attention mechanisms, tokenization, fine-tuning strategies, and practical techniques for working with LLMs in production.

## Transformer Architecture

### Overview

The transformer (Vaswani et al., 2017) replaced recurrence with self-attention, enabling massive parallelization and capturing long-range dependencies.

```
Input tokens → Embedding + Positional Encoding
                        │
              ┌─────────▼──────────┐
              │   Encoder Stack     │    (BERT, encoder-only)
              │   N × [Self-Attn   │
              │        + FFN]       │
              └─────────┬──────────┘
                        │
              ┌─────────▼──────────┐
              │   Decoder Stack     │    (GPT, decoder-only)
              │   N × [Masked Attn │
              │        + Cross-Attn │    (Cross-attention only in encoder-decoder)
              │        + FFN]       │
              └─────────┬──────────┘
                        │
              Linear + Softmax → Output probabilities
```

### Architecture Variants

| Variant | Examples | Self-Attention Type | Best For |
|---------|----------|-------------------|----------|
| **Encoder-only** | BERT, RoBERTa, DeBERTa | Bidirectional | Classification, NER, embeddings |
| **Decoder-only** | GPT, LLaMA, PaLM, Gemini | Causal (left-to-right) | Text generation, chat, code |
| **Encoder-decoder** | T5, BART, mT5 | Bidirectional encoder + causal decoder | Translation, summarization |

### Key Components

**Embedding layer:** Converts token IDs to dense vectors (dimension `d_model`, typically 768–4096).

**Positional encoding:** Injects sequence order information since self-attention is position-agnostic.
- Original: sinusoidal functions of position and dimension
- Modern: learned positional embeddings, RoPE (Rotary Position Embeddings), ALiBi

**Feed-forward network (FFN):** Two linear layers with activation in between, applied independently to each position. Typically 4× `d_model` hidden dimension.

**Layer normalization:** Applied before or after each sub-layer (pre-norm is more stable for deep models).

**Residual connections:** `output = LayerNorm(x + SubLayer(x))` — enables training deep networks.

## Attention Mechanisms

### Scaled Dot-Product Attention

```
Attention(Q, K, V) = softmax(QK^T / √d_k) V
```

```
Q (query)  ──┐
K (key)    ──┼── QK^T / √d_k → softmax → attention weights
V (value)  ──┘                                    │
                                          weighted sum of V
```

- **Q (Query):** "What am I looking for?"
- **K (Key):** "What do I contain?"
- **V (Value):** "What information do I provide?"
- **Scaling (√d_k):** Prevents dot products from growing too large, which would push softmax into saturated regions with tiny gradients

### Multi-Head Attention

Instead of one attention function, project Q, K, V into `h` heads, attend in parallel, concatenate.

```
MultiHead(Q, K, V) = Concat(head_1, ..., head_h) W^O
  where head_i = Attention(QW_i^Q, KW_i^K, VW_i^V)
```

- Each head attends to different aspects (syntax, semantics, positional relationships)
- Typical: 12 heads for base models, 32-96+ for large models
- Head dimension = `d_model / num_heads` (e.g., 768/12 = 64)

### Attention Variants

| Variant | How It Works | Benefit |
|---------|-------------|---------|
| **Causal (masked)** | Mask future positions in attention matrix | Autoregressive generation |
| **Cross-attention** | Q from decoder, K/V from encoder | Encoder-decoder models (translation) |
| **Multi-query attention (MQA)** | Single K/V head shared across all Q heads | Faster inference, less KV cache |
| **Grouped-query attention (GQA)** | K/V heads shared across groups of Q heads | Balance between MHA and MQA |
| **Flash Attention** | IO-aware exact attention, fused kernel | 2-4x faster, memory-efficient |
| **Sliding window** | Attend only to nearby tokens (window size) | Efficient for long sequences (Mistral) |

## Tokenization

### Strategies

| Method | How It Works | Used By |
|--------|-------------|---------|
| **BPE (Byte-Pair Encoding)** | Iteratively merge most frequent character pairs | GPT-2/3/4, LLaMA, RoBERTa |
| **WordPiece** | Similar to BPE but uses likelihood instead of frequency | BERT, DistilBERT |
| **SentencePiece** | Language-agnostic, treats input as raw Unicode (no pre-tokenization) | T5, LLaMA 1/2, multilingual models |
| **Byte-level BPE** | BPE on raw bytes, no unknown tokens possible | GPT-2, GPT-4 |

### Tokenization Effects on Model Behavior

```
"unhappiness" → ["un", "happiness"]           (compositional)
"ChatGPT"     → ["Chat", "G", "PT"]           (splits rare words)
"123456"      → ["123", "456"]                 (numbers split unpredictably)
```

Key considerations:
- **Vocabulary size trade-off:** Larger vocab = shorter sequences but larger embedding table
- **Multilingual:** SentencePiece handles languages without spaces (Chinese, Japanese)
- **Arithmetic:** LLMs struggle with math partly due to inconsistent number tokenization
- **Typical vocab sizes:** 30K-50K (BERT), 50K-100K (GPT), 128K+ (Gemini)

## Prompt Engineering

### Core Patterns

| Pattern | Description | Example |
|---------|-------------|---------|
| **Zero-shot** | Direct instruction, no examples | "Classify this review as positive or negative" |
| **Few-shot** | Provide examples before the query | "Positive: 'Great product!' / Negative: 'Terrible.' / Classify: ..." |
| **Chain-of-thought (CoT)** | Ask model to reason step by step | "Let's think step by step..." |
| **System prompting** | Set role, constraints, output format | "You are a helpful coding assistant. Respond in JSON." |
| **Self-consistency** | Sample multiple CoT paths, take majority vote | Generate 5 reasoning chains, pick most common answer |

### Best Practices
- Be specific about output format (JSON, markdown, bullet points)
- Include examples for complex tasks (few-shot > zero-shot for formatting)
- Break complex tasks into subtasks (chain prompts)
- Use delimiters to separate instructions from content (`"""`, `###`, XML tags)
- Specify what NOT to do (reduces hallucination)

## Fine-Tuning vs RAG vs In-Context Learning

### Comparison

| Approach | Training Required | Knowledge Source | Latency | Best For |
|----------|------------------|-----------------|---------|----------|
| **Fine-tuning** | Yes (hours-days) | Baked into weights | Low (single forward pass) | Style/behavior change, domain adaptation, consistent formatting |
| **RAG** | No (or minimal) | External retrieval at inference time | Higher (retrieval + generation) | Factual knowledge, frequently updated info, citations needed |
| **In-context learning** | No | Provided in prompt | Low | Quick prototyping, few-shot tasks, dynamic instructions |

### When to Use Each

**Fine-tuning when:**
- You need to change the model's behavior, style, or output format consistently
- Domain-specific language that the base model handles poorly
- You have labeled training data (100s-1000s of examples)
- Latency matters and you can't afford retrieval overhead

**RAG (Retrieval-Augmented Generation) when:**
- Knowledge changes frequently (documentation, product info)
- You need citations or source attribution
- Reducing hallucination is critical
- Knowledge base is too large to fit in context window

**In-context learning when:**
- Quick iteration without training
- Task can be defined with a few examples
- You need maximum flexibility to change behavior at runtime

### RAG Architecture

```
User query → Embedding model → Vector search → Top-K documents
                                                      │
                                          ┌───────────▼───────────┐
                                          │ Prompt = instruction   │
                                          │   + retrieved docs     │
                                          │   + user query         │
                                          └───────────┬───────────┘
                                                      │
                                                LLM generates
                                                 response
```

### Fine-Tuning Approaches

| Method | Parameters Updated | Memory | Best For |
|--------|-------------------|--------|----------|
| **Full fine-tuning** | All | Very high | Maximum quality, enough data |
| **LoRA** | Low-rank adapter matrices | Low (~1-10% of full) | Most common, good quality/efficiency trade-off |
| **QLoRA** | LoRA on quantized model | Very low | Fine-tuning large models on consumer GPUs |
| **Prefix tuning** | Learned prefix tokens | Very low | Simple tasks, limited data |
| **Adapter layers** | Small layers inserted between existing | Low | Multi-task learning |

## Hallucination Mitigation

### Types of Hallucination

| Type | Description | Example |
|------|-------------|---------|
| **Factual** | States incorrect facts confidently | "The Eiffel Tower is in London" |
| **Fabrication** | Invents non-existent entities | Citing a paper that doesn't exist |
| **Inconsistency** | Contradicts itself or the source | Says "yes" then "no" to the same question |
| **Instruction drift** | Ignores constraints over long outputs | Stops using requested format mid-response |

### Mitigation Strategies

| Strategy | How It Helps |
|----------|-------------|
| **RAG** | Ground responses in retrieved evidence |
| **Temperature reduction** | Lower temperature (0.0-0.3) for factual tasks |
| **Self-consistency / voting** | Multiple samples, majority vote reduces random errors |
| **Chain-of-thought** | Step-by-step reasoning catches logical errors |
| **Output validation** | Post-process: check facts against knowledge base, validate JSON structure |
| **Constrained decoding** | Force output to match a schema (JSON mode, function calling) |
| **Fine-tuning on refusals** | Train model to say "I don't know" when uncertain |
| **Source attribution** | Require citations, verify against sources |

## Context Window and Long Sequences

### Limitations

| Model | Context Window | Approximate Token Budget |
|-------|---------------|------------------------|
| GPT-4 Turbo | 128K tokens | ~96K words |
| Claude 3.5 | 200K tokens | ~150K words |
| Gemini 1.5 Pro | 1M+ tokens | ~750K words |
| LLaMA 3 | 8K-128K tokens | Varies by version |

### Working with Long Contexts

| Strategy | How It Works | Trade-off |
|----------|-------------|-----------|
| **Chunking + retrieval (RAG)** | Split documents, retrieve relevant chunks | May miss cross-chunk context |
| **Sliding window** | Process overlapping windows, merge results | Redundant computation |
| **Hierarchical summarization** | Summarize chunks, then summarize summaries | Information loss at each level |
| **Map-reduce** | Process chunks independently, combine results | Good for aggregation, bad for reasoning |
| **Rope scaling / position interpolation** | Extend position embeddings beyond training length | Quality degrades at extreme lengths |

## Embeddings and Vector Search

### Embedding Models

Convert text (or images, audio) to dense vectors that capture semantic meaning.

| Model | Dimensions | Use Case |
|-------|-----------|----------|
| **text-embedding-3-large** (OpenAI) | 3072 | General-purpose text embeddings |
| **Gecko** (Google) | 768 | Vertex AI, Google ecosystem |
| **BGE / E5** (open source) | 768-1024 | Self-hosted, no API dependency |
| **CLIP** (OpenAI) | 512-768 | Multi-modal (text + images) |

### Vector Search Algorithms

| Algorithm | Type | Speed | Accuracy | Memory |
|-----------|------|-------|----------|--------|
| **Brute force (flat)** | Exact | Slow | 100% | Low |
| **IVF (Inverted File)** | Approximate | Fast | 95-99% | Low |
| **HNSW** | Approximate (graph-based) | Very fast | 97-99% | High (in-memory graph) |
| **ScaNN** | Approximate (Google) | Very fast | 97-99% | Medium |
| **Product Quantization** | Compressed | Fast | 90-95% | Very low |

### HNSW (Hierarchical Navigable Small Worlds)

```
Layer 2 (sparse):    A ─────────── D
                     │               │
Layer 1 (medium):    A ── B ──── D ── E
                     │    │      │    │
Layer 0 (dense):     A ─ B ─ C ─ D ─ E ─ F
```

- Multi-layer graph with skip-list-like structure
- Search starts at top layer (few nodes, long jumps) and descends to bottom layer (all nodes, precise)
- Trade-offs: `M` (max connections per node) affects quality vs memory; `ef` (search beam width) affects quality vs speed

### Vector Databases

| Database | Type | Key Features |
|----------|------|-------------|
| **Pinecone** | Managed SaaS | Fully managed, serverless option, metadata filtering |
| **Weaviate** | Open source | Multi-modal, hybrid search (vector + keyword) |
| **ChromaDB** | Open source | Simple API, good for prototyping |
| **pgvector** | PostgreSQL extension | Use existing Postgres, no new infra |
| **Vertex AI Vector Search** | Google managed | Integrated with GCP, ScaNN-based |
