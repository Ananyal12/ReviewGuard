# 🔎 Fake Review Detector

An NLP-based machine learning application that classifies product/service reviews as **likely fake** or **likely genuine**.

## 🚀 Live Demo

[🔗 Try ReviewGuard Live](https://reviewguard-adj22rp.streamlit.app/)

## Features

- TF-IDF text representation with up to 5,000 features
- Three handcrafted writing-style features:
  - Exclamation-mark count
  - Uppercase-character count
  - Word count
- Logistic Regression classifier
- Stratified 80/20 train-test evaluation
- Streamlit web interface
- Prediction confidence display
- Extracted-feature preview
- Reproducible training script

## Model performance

Using `random_state=42` and a stratified 80/20 split, the corrected training pipeline achieves approximately:

- **Accuracy: 91.02%**
- **F1-score: 0.91** for both classes
- Test set: **8,087 reviews**

The TF-IDF vectorizer is fitted only on the training data to avoid test-set leakage.

## How to run

```bash
pip install -r requirements.txt
streamlit run app.py
```

## Retrain the model

```bash
python train_model.py
```

This recreates `model.pkl` and `tfidf.pkl` from `data/raw/fake_reviews.csv`.

## Project workflow

```text
Review entered by user
        ↓
Text preprocessing
        ↓
TF-IDF transformation (5,000 features)
        +
Writing-style features
(exclamation count, uppercase count, word count)
        ↓
Feature combination
        ↓
Logistic Regression
        ↓
Prediction + probability
        ↓
Streamlit result: Likely Fake / Likely Genuine
```

## Important note

A machine-learning prediction is an estimate based on learned patterns. It should not be treated as definitive proof that a review is fake or genuine.
