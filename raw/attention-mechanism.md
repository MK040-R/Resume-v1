# The Attention Mechanism in Deep Learning

## Overview

The attention mechanism is a technique that allows neural networks to focus on
specific parts of the input when producing output. Originally developed for machine
translation (Bahdanau et al., 2014), it became the foundation of the transformer architecture.

## How Attention Works

### Scaled Dot-Product Attention

The core computation involves three learned projections:
- **Query (Q)**: What we're looking for
- **Key (K)**: What each element represents
- **Value (V)**: The actual information each element carries

The attention output for a given query is:
```
Attention(Q, K, V) = softmax(QK^T / √d_k) · V
```

The scaling factor `√d_k` prevents the dot products from becoming too large
in high dimensions, which would push the softmax into regions with very small gradients.

### Multi-Head Attention

Instead of computing a single attention function, multi-head attention runs the
attention mechanism multiple times in parallel with different learned projections:

```
MultiHead(Q, K, V) = Concat(head_1, ..., head_h) · W^O
where head_i = Attention(Q·W_i^Q, K·W_i^K, V·W_i^V)
```

Each "head" can learn to attend to different aspects of the input.
Typical transformer models use 8-16 heads.

## Types of Attention

### Self-Attention
In self-attention, the queries, keys, and values all come from the same sequence.
This allows each token to attend to every other token in the same sequence.
Used in the encoder of transformer models.

### Cross-Attention
Used in the decoder, where queries come from the decoder and keys/values come
from the encoder output. This is how the decoder "reads" the encoded input.

### Causal (Masked) Self-Attention
Used in autoregressive models like GPT. A mask prevents each position from
attending to future positions, ensuring the model can only use past context.

## Attention Patterns

Research has shown that different attention heads learn different roles:
- Some heads track syntactic dependencies (subject-verb agreement)
- Some heads track positional patterns (attending to adjacent tokens)
- Some heads attend to specific semantic relationships

## Computational Complexity

Self-attention has O(n²) complexity with respect to sequence length n,
which is a limitation for very long sequences. This has motivated research into:
- Sparse attention patterns (Longformer, BigBird)
- Linear attention approximations
- Flash Attention (memory-efficient exact attention)

## Key Papers
- Bahdanau et al. (2014): "Neural Machine Translation by Jointly Learning to Align and Translate"
- Vaswani et al. (2017): "Attention is All You Need"
- Devlin et al. (2018): "BERT: Pre-training of Deep Bidirectional Transformers"
