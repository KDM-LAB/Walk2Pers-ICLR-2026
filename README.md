# Walk2Pers

Code and notebooks for the paper:

**Beyond Markovian Drifts: Action-Biased Geometric Walks with Memory for Personalized Summarization**

This repository contains a small, notebook-driven pipeline to (i) sanity-check / preprocess personalized summarization data, (ii) build text embeddings (e.g., T5 encodings), (iii) train Walk2Pers model variants, and (iv) run evaluation / testing.

> Note: This repo is organized primarily around Jupyter notebooks (see **Notebooks** below). If you prefer scripts, you can convert notebooks to `.py` via `jupyter nbconvert --to script`.

---

## What is Walk2Pers?

Walk2Pers operationalizes the **Structured Walk Hypothesis (SWH)** for personalized summarization: user preference evolves over time and should be modeled beyond short-memory / purely Markov drift assumptions. The method updates a user preference state using **action-conditioned steps** (direction + magnitude) and **dual memory lanes** (reinforce vs. suppress), designed specifically for summarization.

---

## Repository Contents (high level)

- `README.md` — you’re reading it
- `check_data.ipynb` — dataset inspection / sanity checks
- `test_data.ipynb` — quick testing utilities on prepared data
- `embedding_T5_encoding.ipynb` — embedding generation (T5-based encoding)
- `w2p_model_test.ipynb` — model loading + evaluation / inference tests
- `w2p_model_training_lmhead+nlayers.ipynb` — training notebook (variant)
- `w2p_model_training_lmhead_only (1).ipynb` — training notebook (variant)

---

## Setup

### 1) Environment
This codebase assumes a standard Python ML stack.

Recommended:
- Python 3.9+ (3.10/3.11 typically fine)
- PyTorch (CUDA optional)
- HuggingFace `transformers`, `datasets`
- `numpy`, `pandas`, `scikit-learn`, `tqdm`, `matplotlib`

If you already have your own environment management:
- Use `pip` / `conda` as usual and install missing packages based on notebook import errors.

Example (minimal, adjust as needed):
```bash
pip install torch transformers datasets numpy pandas scikit-learn tqdm matplotlib jupyter
