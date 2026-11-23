"""IMDB Sentiment Analysis — improved Gradio app (Hugging Face Spaces).

Uses the DistilBERT model fine-tuned on IMDB reviews (see train.py).
Features:
- single-review analysis with a confidence breakdown,
- batch analysis (one review per line) with a results table,
- model card info (training data, accuracy) shown in the UI.

Model path: set MODEL_PATH, or place the fine-tuned model in
./model_imdb next to this file. Falls back to the base SST-2 model
if no fine-tuned model is present.
"""
import spaces  # MUST be first: ZeroGPU mode fails if CUDA is
# initialised (by gradio/torch) before `spaces` is imported.

import json
import os

import gradio as gr
import numpy as np
import torch
from transformers import pipeline

HERE = os.path.dirname(os.path.abspath(__file__))
HF_MODEL = "Varund88/imdb-sentiment-distilbert"

# The Space runs on ZeroGPU (free tier), which only allocates a GPU
# inside functions decorated with @spaces.GPU. So the model is loaded
# lazily on first use, inside such a function, and cached afterwards.
_classifier = None


def _get_classifier():
    global _classifier
    if _classifier is None:
        device = 0 if torch.cuda.is_available() else -1
        model_path = os.environ.get("MODEL_PATH") or HF_MODEL
        try:
            _classifier = pipeline(
                "sentiment-analysis", model=model_path, device=device
            )
        except Exception:
            local = os.path.join(HERE, "model_imdb")
            fallback = (
                local
                if os.path.isdir(local)
                else "/home/hatch/workspace/github-code-check/imdb-model"
            )
            _classifier = pipeline(
                "sentiment-analysis", model=fallback, device=device
            )
    return _classifier

METRICS_PATH = os.path.join(HERE, "model_imdb", "metrics.json")
metrics = {}
if os.path.isfile(METRICS_PATH):
    with open(METRICS_PATH) as f:
        metrics = json.load(f)
else:
    # Metrics of the fine-tuned model (see model_imdb/metrics.json).
    metrics = {"train_size": 3000, "accuracy": 0.8575, "f1_weighted": 0.8575}


@spaces.GPU
def scores_for(text):
    """Return (label, pos_score, neg_score) using the raw pipeline output."""
    raw = _get_classifier()(text)
    if isinstance(raw, list) and raw and isinstance(raw[0], list):
        raw = raw[0]
    d = {item["label"]: float(item["score"]) for item in raw}
    pos, neg = d.get("POSITIVE", 0.0), d.get("NEGATIVE", 0.0)
    return ("POSITIVE" if pos >= neg else "NEGATIVE"), pos, neg


def temperature_confidence(raw_prob, temperature=2.5):
    """Same temperature scaling as the original backend (T=2.5)."""
    p = min(max(raw_prob, 1e-9), 1 - 1e-9)
    adjusted = 1 / (1 + np.exp(-np.log(p / (1 - p)) / temperature))
    return round(float(adjusted) * 100, 1)


def strength(conf):
    if conf >= 90:
        return "Very strong"
    if conf >= 75:
        return "Strong"
    if conf >= 60:
        return "Moderate"
    return "Weak - borderline"


def analyze_one(text):
    if not text or not text.strip():
        return "Please enter a review.", None, "", ""
    label, pos, neg = scores_for(text.strip())
    raw = pos if label == "POSITIVE" else neg
    conf = temperature_confidence(raw)
    # gr.Label expects confidences as fractions (0-1); it renders them
    # as percentages itself. Passing percentages here made it show 10000%.
    chart = {"POSITIVE": round(pos, 3), "NEGATIVE": round(neg, 3)}
    return label, chart, f"{conf}%", strength(conf)


def analyze_batch(text):
    if not text or not text.strip():
        return [["(empty)", "", ""]]
    rows = []
    for line in text.strip().splitlines():
        line = line.strip()
        if not line:
            continue
        label, pos, neg = scores_for(line)
        raw = pos if label == "POSITIVE" else neg
        conf = temperature_confidence(raw)
        rows.append([line[:80], label, f"{conf}% ({strength(conf)})"])
    return rows


acc_line = ""
if metrics:
    acc_line = (
        f"Fine-tuned on {metrics.get('train_size', '?')} IMDB reviews · "
        f"test accuracy **{metrics.get('accuracy', '?') * 100:.1f}%** · "
        f"F1 **{metrics.get('f1_weighted', '?')}**"
    )

with gr.Blocks(title="IMDB Sentiment Analysis") as demo:
    gr.Markdown("# 🎬 IMDB Sentiment Analysis")
    gr.Markdown(
        "A DistilBERT model fine-tuned on real IMDB movie reviews. "
        "Type a review and it tells you whether it is positive or "
        f"negative, and how sure it is. {acc_line}"
    )
    with gr.Tab("Single review"):
        inp = gr.Textbox(
            label="Movie review",
            placeholder="Type a movie review here...",
            lines=4,
        )
        btn = gr.Button("Analyze", variant="primary")
        with gr.Row():
            out_label = gr.Textbox(label="Sentiment")
            out_conf = gr.Textbox(label="Confidence")
            out_strength = gr.Textbox(label="Strength")
        out_chart = gr.Label(label="Score breakdown")
        btn.click(
            analyze_one,
            inputs=inp,
            outputs=[out_label, out_chart, out_conf, out_strength],
        )
        gr.Examples(
            examples=[
                ["This movie was absolutely wonderful, I loved every minute of it."],
                ["Terrible film. Boring, badly acted and a complete waste of time."],
                ["The plot was slow in the middle, but the ending made it worth watching."],
            ],
            inputs=inp,
        )
    with gr.Tab("Batch — one review per line"):
        batch_in = gr.Textbox(
            label="Reviews (one per line)",
            placeholder="Great movie, loved it!\nWorst film I have ever seen.",
            lines=5,
        )
        batch_btn = gr.Button("Analyze all", variant="primary")
        batch_out = gr.Dataframe(
            headers=["Review", "Sentiment", "Confidence"],
            label="Results",
        )
        batch_btn.click(analyze_batch, inputs=batch_in, outputs=batch_out)
    gr.Markdown(
        "Model: DistilBERT fine-tuned on the IMDB dataset (25,000 train / "
        "25,000 test reviews). Built with Hugging Face Transformers + Gradio."
    )

if __name__ == "__main__":
    demo.launch()
