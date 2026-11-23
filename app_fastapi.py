from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from transformers import pipeline
import numpy as np

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
    allow_credentials=False
)

classifier = pipeline(
    "sentiment-analysis",
    model="distilbert-base-uncased-finetuned-sst-2-english",
    return_all_scores=True
)

class Review(BaseModel):
    text: str

def get_real_confidence(raw_output):
    # Flatten whatever structure comes in
    print("RAW OUTPUT:", raw_output)
    print("TYPE:", type(raw_output))

    # Handle all possible formats
    if isinstance(raw_output, list):
        if isinstance(raw_output[0], list):
            # Double nested list [[{...}, {...}]]
            scores = raw_output[0]
        elif isinstance(raw_output[0], dict):
            # Single list [{...}, {...}]
            scores = raw_output
        else:
            scores = raw_output
    else:
        scores = raw_output

    print("SCORES:", scores)

    # Build score dictionary
    score_dict = {}
    for item in scores:
        score_dict[item["label"]] = item["score"]

    print("SCORE DICT:", score_dict)

    neg_score = score_dict.get("NEGATIVE", 0)
    pos_score = score_dict.get("POSITIVE", 0)

    if pos_score > neg_score:
        label = "POSITIVE"
        raw = pos_score
    else:
        label = "NEGATIVE"
        raw = neg_score

    # Temperature scaling
    temperature = 2.5
    adjusted = 1 / (1 + np.exp(-np.log(raw / (1 - raw + 1e-9)) / temperature))
    confidence = round(float(adjusted) * 100, 1)

    return label, confidence

def get_interpretation(score):
    if score >= 90:
        return "Very strong"
    elif score >= 75:
        return "Strong"
    elif score >= 60:
        return "Moderate"
    else:
        return "Weak - borderline review"

@app.post("/predict")
def predict(review: Review):
    raw_output = classifier(review.text[:512])
    label, confidence = get_real_confidence(raw_output)
    return {
        "label": label,
        "score": confidence,
        "interpretation": get_interpretation(confidence)
    }

@app.get("/")
def root():
    return {"status": "running"}