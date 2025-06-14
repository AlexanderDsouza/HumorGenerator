# HumorGenerator
This repository implements a two-stage humor enhancement system for transforming neutral sentences into humorous ones. It combines a fine-tuned text-to-text transformer (T5) with a BERT-based humor classifier (H(x)) for sentence filtering and evaluation.

## Project Overview
Humor is a complex and subjective aspect of human communication. This project attempts to generate funnier variations of neutral sentences using:

1. Humor Classification Model H(x) — built using BERT embeddings and trained with both Logistic Regression and a Neural Net (MLP).
2. Sentence Pair Generation — using GPT-4 Turbo to produce (neutral, humorous) sentence pairs.
3. Fine-Tuned Humor Transformer — trained on high-confidence humorous pairs filtered by H(x) using the T5 architecture.

## Directory Structure
HumorGenerator/ \
├── Humor Classifier.ipynb # Builds and evaluates the BERT-based classifier H(x) \
├── LocalSentencePairGenerator.ipynb # Uses GPT-4 to generate (neutral, funny) sentence pairs \
├── T5Trainer.ipynb # Trains T5-Large model on local sentence pairs from GPT-4 and H(x) \
├── LogisticRegression_model.pkl # Saved logistic regression model for H(x) \
├── NeuralNet_model.pkl # Saved MLP (Neural Net) model for H(x) \
├── X_train.pickle # Training features (sentence embeddings) \
├── y_train.pickle # Training labels (0=not funny, 1=funny) \
├── X_test.pickle # Test features \
├── y_test.pickle # Test labels \
└── README.md

## Installation
Clone the repository and install dependencies:
```
git clone https://github.com/AlexanderDsouza/HumorGenerator.git
cd HumorGenerator
pip install -r requirements.txt
```

If requirements.txt is missing, install manually:
`pip install transformers torch scikit-learn pandas openai`

## How to Run
1. Train Humor Classifier
Run Humor Classifier.ipynb to train or load a pre-trained H(x) model based on BERT embeddings.

2. Generate Sentence Pairs
Use LabeledSentencePairGenerator.ipynb to generate (neutral, humorous) sentence pairs using GPT-4 Turbo.

You'll need an OpenAI API key to run the generator.

3. Train Humor Style Transfer Model
Not included in this repo, but described in the report: fine-tune a T5 or Flan-T5 model on your generated dataset.

## Results Summary
| Model	| Accuracy | Precision | Recall | F1 Score | Specificity |
| ----- | -------- | --------- | ------ | -------- | ----------- |
| Logistic Regression | 95.62% | 92.38% | 94.99% | 93.67% | 95.91% |
| Neural Net (MLP) |95.51% | 91.72% | 95.48% | 93.56% | 95.50% |

These models were trained on filtered jokes from a Kaggle dataset and achieved high performance on binary humor classification.

## Key Features
- Lightweight pipeline using BERT + T5.
- Generated data filtered via self-trained classifier H(x).
- Supports fine-tuning and analysis of humor mechanics.
- Exploratory metrics including perplexity and pattern detection.

## Future Work
- Enable end-to-end BERT fine-tuning for richer representation learning.
- Explore RLHF using H(x) as a reward function.
- Collect human evaluations or crowd-sourced ratings of humor.
- Extend to multimodal humor generation (text + image).
- Improve style diversity via curriculum learning or GPT-4-level models.

## Citation
If you use this work, consider citing our project report (coming soon).
