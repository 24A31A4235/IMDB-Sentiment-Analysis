# IMDB Sentiment Analysis

A movie-review sentiment classifier built by **fine-tuning DistilBERT on the
IMDB dataset** — not just calling a pre-trained model. Type a review and it
tells you whether it is positive or negative, how confident it is, and why.

**Live demo:** *(Hugging Face Spaces link — added after deploy)*

## What it does

- **Single review:** paste one review, get Positive/Negative, a confidence
  score (temperature-scaled, like the original backend) and a score breakdown.
- **Batch mode:** paste several reviews (one per line) and get a results table.
- **Trained, not just loaded:** the model was fine-tuned on real IMDB
  reviews, and its accuracy is measured on reviews it has never seen.

## Model & training

| | |
|---|---|
| Base model | DistilBERT (`distilbert-base-uncased`, fine-tuned from the SST-2 checkpoint) |
| Dataset | IMDB — 25,000 train / 25,000 test movie reviews, balanced pos/neg |
| Training set | 3,000 reviews (1,500 pos + 1,500 neg), 2 epochs |
| Validation | 1,000 held-out reviews |
| Test set | 2,000 held-out reviews (never seen in training) |
| Frozen layers | embeddings + bottom 4 of 6 transformer layers (memory-efficient) |
| Max sequence length | 128 tokens |
| Batch / LR | 8 (grad accum 2 → effective 16) / 2e-5, AdamW |

### Results (held-out IMDB test set, 2,000 reviews)

| Metric | Value |
|---|---|
| Accuracy | **85.75%** |
| Weighted F1 | **0.8575** |

Confusion matrix and per-class precision/recall are in
`model_imdb/metrics.json` after training.

> The previous version of this project only *loaded* a pre-trained SST-2
> model and never trained anything. This version fine-tunes on IMDB and
> reports honest, measured accuracy — that is the difference a recruiter
> can check.

## How it works

1. `train.py` loads IMDB reviews, tokenizes them and fine-tunes DistilBERT
   with the Hugging Face `Trainer` (see `train.py` — frozen bottom layers,
   gradient accumulation, eval each epoch).
2. `app.py` (Gradio) loads the fine-tuned model and serves it: single-review
   and batch analysis, with a Positive/Negative score breakdown.
3. The original FastAPI backend (`app_fastapi.py`) is kept for API use.

## Project structure

```
IMDB-Sentiment-Analysis/
├── train.py            # fine-tune DistilBERT on IMDB (+ evaluation)
├── app.py              # Gradio UI (single + batch analysis)
├── app_fastapi.py      # original FastAPI backend (/predict)
├── model_imdb/         # fine-tuned model + tokenizer + metrics.json
├── requirements.txt
└── README.md
```

## Run it locally

```bash
pip install -r requirements.txt
# train (takes ~1h on CPU; skip if model_imdb/ is already there)
python train.py --train-size 3000 --epochs 2 --out model_imdb
# UI
python app.py
# or the API
uvicorn app_fastapi:app --reload
```

## Deploy (Hugging Face Spaces)

Upload `app.py` + `requirements.txt` + `model_imdb/` to a new Gradio Space.
The Space installs the dependencies and serves the fine-tuned model.

## Requirements

- Python 3.9+
- `transformers`, `torch`, `gradio`, `numpy`, `scikit-learn`, `accelerate`
- (training) ~1 GB RAM free, CPU only — no GPU needed for this size
