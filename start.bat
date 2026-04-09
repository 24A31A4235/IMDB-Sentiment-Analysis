@echo off
echo Starting Sentiment Analysis App...
echo.
start cmd /k "cd C:\Users\VARUN\OneDrive\Desktop\IMDB_Project && python -m uvicorn app:app --reload"
timeout /t 5 /nobreak
start cmd /k "cd C:\Users\VARUN\OneDrive\Desktop\IMDB_Project\sentiment-app && npm run dev"
start http://localhost:5173