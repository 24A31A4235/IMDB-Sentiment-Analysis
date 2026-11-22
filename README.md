# 🎬 IMDB Sentiment Analysis

A full-stack ML web app that classifies movie reviews as Positive or Negative.

## 🔥 Tech Stack
- **Model:** DistilBERT (Hugging Face Transformers)
- **Dataset:** IMDB 50k reviews
- **Backend:** FastAPI (Python)
- **Frontend:** React + Vite
- **Accuracy:** 93%+

## 🚀 How to Run

### Backend
cd IMDB_Project
pip install fastapi uvicorn transformers torch
python -m uvicorn app:app --reload

### Frontend
cd IMDB_Project/sentiment-app
npm install
npm run dev

Open localhost:5173 in browser.

## 📊 Features
- Real-time sentiment prediction
- Confidence score with temperature scaling
- Confidence bar visualization
- Very Strong / Strong / Moderate / Weak interpretation

## 🧠 Model Details
- Base model: distilbert-base-uncased-finetuned-sst-2-english
- Fine-tuned on IMDB dataset (25k train, 25k test)
- Accuracy: 93%+
- Temperature scaling applied for realistic confidence scores
