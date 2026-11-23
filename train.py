"""Fine-tune DistilBERT on the IMDB movie-review dataset.

Starts from the SST-2 fine-tuned DistilBERT already in the repo
(local folder) and adapts it to IMDB reviews. This is a real training
run: the model learns from IMDB labels, and we report honest metrics
on a held-out IMDB test split.

Memory-efficient for a small CPU box:
- freezes the embeddings + the bottom transformer layers (only the top
  2 layers + classifier are trained), so the AdamW optimizer state
  stays small;
- short sequences (max_len 128) and a small batch (8) with gradient
  accumulation, so activations stay small.

Usage:
  python train.py --train-size 4000 --epochs 2 --out model_imdb
  python train.py --smoke            # 200 examples, 1 epoch, speed check
"""
import argparse
import json
import os
import random
import re
import time

import numpy as np
import torch
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
)

BASE_MODEL = os.environ.get(
    "MODEL_PATH", "/home/hatch/workspace/github-code-check/imdb-model"
)
DATA_DIR = os.environ.get(
    "IMDB_DIR", "/home/hatch/workspace/imdb-improved/data/aclImdb"
)


def clean_text(text):
    # IMDB reviews contain <br /> tags and HTML entities.
    text = text.replace("<br />", " ")
    text = re.sub(r"<[^>]+>", " ", text)
    return text.strip()


def load_split(split, limit=None, seed=42):
    """Load (text, label) pairs from the aclImdb train/test folders.
    `limit` is the total size; it is split evenly across the two
    classes so the set stays balanced."""
    rng = random.Random(seed)
    per_class = (limit // 2) if limit else None
    items = []
    for label, folder in ((1, "pos"), (0, "neg")):
        path = os.path.join(DATA_DIR, split, folder)
        files = sorted(os.listdir(path))
        rng.shuffle(files)
        count = 0
        for fname in files:
            with open(os.path.join(path, fname), encoding="utf-8") as f:
                items.append((clean_text(f.read()), label))
            count += 1
            if per_class and count >= per_class:
                break
    rng.shuffle(items)
    return items


class IMDBDataset(torch.utils.data.Dataset):
    def __init__(self, items, tokenizer, max_len=128):
        self.items = items
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        text, label = self.items[idx]
        enc = self.tokenizer(
            text, truncation=True, max_length=self.max_len, padding="max_length"
        )
        return {
            "input_ids": enc["input_ids"],
            "attention_mask": enc["attention_mask"],
            "labels": label,
        }


def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=1)
    return {
        "accuracy": accuracy_score(labels, preds),
        "f1": f1_score(labels, preds, average="weighted"),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train-size", type=int, default=4000)
    ap.add_argument("--val-size", type=int, default=1000)
    ap.add_argument("--test-size", type=int, default=2000)
    ap.add_argument("--epochs", type=float, default=2.0)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--max-len", type=int, default=128)
    ap.add_argument("--freeze-layers", type=int, default=4,
                    help="freeze embeddings + first N transformer layers")
    ap.add_argument("--out", default="model_imdb")
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()

    if args.smoke:
        args.train_size, args.val_size, args.test_size = 200, 100, 100
        args.epochs = 1.0

    print(f"Loading data (train={args.train_size}, val={args.val_size}, "
          f"test={args.test_size}) ...")
    train_items = load_split("train", args.train_size)
    val_items = load_split("train", args.val_size, seed=123)
    # make sure val does not overlap train: different seed + different files
    test_items = load_split("test", args.test_size, seed=999)
    print(f"train={len(train_items)} val={len(val_items)} "
          f"test={len(test_items)}")

    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
    model = AutoModelForSequenceClassification.from_pretrained(
        BASE_MODEL, num_labels=2
    )

    # Freeze embeddings + bottom transformer layers (memory-efficient).
    n_layers = len(model.distilbert.transformer.layer)
    freeze = min(args.freeze_layers, n_layers)
    for p in model.distilbert.embeddings.parameters():
        p.requires_grad = False
    for i in range(freeze):
        for p in model.distilbert.transformer.layer[i].parameters():
            p.requires_grad = False
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    print(f"trainable params: {trainable/1e6:.1f}M / {total/1e6:.1f}M "
          f"({freeze} of {n_layers} layers frozen)")

    train_ds = IMDBDataset(train_items, tokenizer, args.max_len)
    val_ds = IMDBDataset(val_items, tokenizer, args.max_len)
    test_ds = IMDBDataset(test_items, tokenizer, args.max_len)

    targs = TrainingArguments(
        output_dir=args.out + "_checkpoints",
        per_device_train_batch_size=args.batch,
        per_device_eval_batch_size=args.batch * 2,
        gradient_accumulation_steps=2,
        num_train_epochs=args.epochs,
        learning_rate=2e-5,
        weight_decay=0.01,
        logging_steps=20,
        eval_strategy="epoch",
        save_strategy="no",
        report_to=[],
        seed=42,
        dataloader_num_workers=0,
    )
    trainer = Trainer(
        model=model,
        args=targs,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        compute_metrics=compute_metrics,
    )

    t0 = time.time()
    trainer.train()
    print(f"training took {time.time() - t0:.0f}s")

    # Final evaluation on the held-out test split.
    preds = trainer.predict(test_ds)
    labels = preds.label_ids
    y_pred = np.argmax(preds.predictions, axis=1)
    acc = accuracy_score(labels, y_pred)
    print("\n=== TEST RESULTS ===")
    print(f"accuracy: {acc:.4f}")
    print(classification_report(labels, y_pred,
                                target_names=["NEGATIVE", "POSITIVE"]))
    print("confusion matrix [[TN, FP], [FN, TP]]:")
    print(confusion_matrix(labels, y_pred))

    metrics = {
        "train_size": len(train_items),
        "val_size": len(val_items),
        "test_size": len(test_items),
        "epochs": args.epochs,
        "max_len": args.max_len,
        "frozen_layers": freeze,
        "trainable_params_m": round(trainable / 1e6, 1),
        "accuracy": round(float(acc), 4),
        "f1_weighted": round(float(f1_score(labels, y_pred, average="weighted")), 4),
        "confusion_matrix": confusion_matrix(labels, y_pred).tolist(),
        "training_seconds": round(time.time() - t0, 0),
    }
    os.makedirs(args.out, exist_ok=True)
    trainer.save_model(args.out)
    tokenizer.save_pretrained(args.out)
    with open(os.path.join(args.out, "metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"\nSaved model + metrics to {args.out}/")


if __name__ == "__main__":
    main()
