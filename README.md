# Marathi-English Transliteration with Attention Mechanisms

## Project Overview
This project implements a sequence-to-sequence model with attention mechanisms for transliterating between Latin script (English) and Devanagari script (Marathi). The system handles character-level conversion with visualizations to interpret attention patterns.

## Key Features
- **Dual Architecture**: Vanilla seq2seq and attention-based models
- **Visual Interpretability**: Attention heatmaps for model decisions
- **Hyperparameter Optimization**: WandB integration for automated sweeps
- **Production-Ready**: Model saving/loading and prediction export
- **Comparative Analysis**: Benchmarks attention vs vanilla approaches

## Model Architectures
1. **Vanilla Seq2Seq**  
   LSTM/GRU encoder-decoder with fixed-length vector bottleneck

2. **Attention Model**  
   Bahdanau-style additive attention with context vector augmentation

## Dataset
Uses the [Dakshina Dataset](https://github.com/google-research-datasets/dakshina) for Marathi:

| File                      | Samples | Purpose          |
|---------------------------|---------|------------------|
| `mr.translit.sampled.train.tsv` | 10,000  | Training         |
| `mr.translit.sampled.dev.tsv`   | 1,000   | Validation       |
| `mr.translit.sampled.test.tsv`  | 1,000   | Testing          |

Sample format:

घर ghar 3
मराठी marathi 2
