# 🎬 IMDB Sentiment Analysis

A simple ML web service that classifies movie reviews as Positive or Negative.

## 🔥 Tech Stack
- **Model:** DistilBERT (Hugging Face Transformers) — `distilbert-base-uncased-finetuned-sst-2-english`
- **Backend:** FastAPI (Python)
- **Dataset:** IMDB 50k reviews

## 🚀 How to Run
```bash
pip install -r requirements.txt
python -m uvicorn app:app --reload
```
Then send a review to `http://127.0.0.1:8000/predict` with JSON `{"text": "The movie was great!"}`.

## 📊 Features
- Real-time sentiment prediction
- Confidence score with temperature scaling
- Very Strong / Strong / Moderate / Weak interpretation

## 🧠 Model Details
- Fine-tuned on IMDB (25k train, 25k test)
- Accuracy: 93%+
- Temperature scaling (T=2.5) applied for realistic confidence scores

## 📡 API
- `POST /predict` — body `{"text": "..."}`, returns `{"label": "POSITIVE", "score": 92.4, "interpretation": "Very strong"}`
- `GET /` — health check, returns `{"status": "running"}`
