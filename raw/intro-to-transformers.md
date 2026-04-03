# Introduction to Transformers

Transformers are a type of deep learning model architecture introduced in the 2017 paper
"Attention is All You Need" by Vaswani et al. They have revolutionized natural language
processing and are now the dominant architecture for most NLP tasks.

## Background

Before transformers, sequential models like RNNs and LSTMs were the standard for NLP.
These architectures processed tokens one at a time, which made parallelization difficult
and caused issues with long-range dependencies.

## Core Architecture

The transformer consists of an encoder and a decoder, each made up of stacked layers.
Each layer contains two main sub-components:

1. **Multi-Head Self-Attention**: Allows the model to attend to different positions of the
   input sequence simultaneously, capturing relationships between tokens regardless of distance.

2. **Feed-Forward Network**: A position-wise fully connected network applied identically
   to each position.

Both sub-components use residual connections and layer normalization.

## Key Innovation: Attention Mechanism

The self-attention mechanism computes a weighted sum of all input representations.
For each token, it computes Query (Q), Key (K), and Value (V) vectors.

The attention score is computed as:
```
Attention(Q, K, V) = softmax(QK^T / sqrt(d_k)) * V
```

Where `d_k` is the dimension of the key vectors (used for scaling to prevent vanishing gradients).

## Positional Encoding

Since transformers process all tokens in parallel, they need a way to encode position.
This is done by adding sinusoidal positional encodings to the input embeddings.

## Advantages Over RNNs

- **Parallelization**: All tokens processed simultaneously during training
- **Long-range dependencies**: Attention can directly connect distant tokens
- **Interpretability**: Attention weights can be visualized

## Applications

Transformers are used in:
- Language models (GPT series, BERT, T5)
- Machine translation
- Text summarization
- Question answering
- Code generation
