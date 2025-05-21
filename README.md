# DA6401 Assignment 03 - Marathi-English Transliteration with Attention Mechanisms

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


## Project Structure
## Project Structure
transliteration-project/\
├── notebooks/\
│└── da6401-a3.ipynb # Complete training and analysis notebook\
├── predictions_attention/ # Attention model predictions\
│└── test_predictions.txt # Format: input,prediction,target\
├── predictions_vanilla/ # Vanilla model predictions\
|└── test_predictions.txt\
├── attention_train.py # Attention model training script\
├── vanilla_train.py # Vanilla model training script\
└── README.md

## Key Features
- **Ready-to-Run Scripts**:
  - `vanilla_train.py`: Pre-configured seq2seq LSTM training
  - `attention_train.py`: Pre-configured attention model training
- **Complete Notebook**: Includes:
  - Data loading and preprocessing
  - Model training and evaluation
  - Attention visualization
  - Error analysis

## Usage Instructions

### Running Training Scripts
```bash
# Run vanilla model training
python vanilla_train.py

# Run attention model training
python attention_train.py
