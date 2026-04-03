# BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding

## Overview

BERT (Bidirectional Encoder Representations from Transformers) was introduced by Google
in 2018 (Devlin et al.). It fundamentally changed how NLP models are trained by introducing
a highly effective pre-training approach.

## Key Contribution: Bidirectional Pre-training

Previous language models (like GPT-1) used left-to-right training — each token only
attends to tokens that came before it. BERT trains bidirectionally, meaning each token
can attend to all other tokens in both directions simultaneously.

This is achieved using two pre-training objectives:

### 1. Masked Language Model (MLM)
- 15% of input tokens are randomly masked
- The model must predict the original masked tokens
- This forces the model to build rich contextual representations
- Of the 15% masked: 80% replaced with [MASK], 10% random word, 10% unchanged

### 2. Next Sentence Prediction (NSP)
- Model receives two sentences A and B
- 50% of the time B is the actual next sentence; 50% is a random sentence
- Model must predict IsNext or NotNext
- (Note: Later research showed NSP may not be necessary — RoBERTa removed it)

## Architecture

BERT uses only the encoder part of the transformer:
- BERT-Base: 12 layers, 768 hidden size, 12 attention heads, 110M parameters
- BERT-Large: 24 layers, 1024 hidden size, 16 attention heads, 340M parameters

## Special Tokens

- `[CLS]`: Added at the beginning; its representation is used for classification tasks
- `[SEP]`: Separates sentence pairs
- `[MASK]`: Replaces masked tokens during pre-training

## Fine-tuning Approach

BERT's power comes from its fine-tuning paradigm:
1. Pre-train on large unlabeled corpus (Wikipedia + BookCorpus)
2. Fine-tune on task-specific labeled data with minimal architectural changes

Tasks BERT can be fine-tuned for:
- Sentence classification (add classifier on [CLS] token)
- Token classification / NER (classify each token)
- Question answering (predict start/end token positions)
- Sentence pair tasks (entailment, similarity)

## Impact and Variants

BERT established the "pre-train then fine-tune" paradigm for NLP.

Key variants:
- **RoBERTa**: More data, no NSP, dynamic masking — improved performance
- **DistilBERT**: 40% smaller, 60% faster, retains 97% of performance
- **ALBERT**: Parameter sharing to reduce model size
- **DeBERTa**: Disentangled attention, improved positional encoding

## Limitations

- Input limited to 512 tokens (quadratic attention cost)
- Encoder-only: not designed for generation tasks
- Fine-tuning requires labeled data per task
- Pre-training is computationally expensive

## Connection to Attention

BERT uses multi-head self-attention throughout. Analysis of BERT's attention heads
has revealed interpretable patterns:
- Heads that attend to delimiter tokens ([SEP], [CLS])
- Heads tracking syntactic structure
- Heads sensitive to positional information

This connects deeply to the general attention mechanism research literature.
