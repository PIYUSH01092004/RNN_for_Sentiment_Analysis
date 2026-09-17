import os
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel
from typing import List, Optional
import uvicorn

from sentiment_engine import SentimentEngine

app = FastAPI(
    title="RNN Sentiment Analysis API",
    description="PyTorch SimpleRNN Sentiment Classifier trained on IMDB Dataset",
    version="1.0.0"
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize Engine
engine = SentimentEngine()

class ReviewRequest(BaseModel):
    text: str

class BatchReviewRequest(BaseModel):
    reviews: List[str]

@app.get("/api/health")
def health_check():
    return {"status": "ok", "model_loaded": engine.model is not None}

@app.get("/api/model-info")
def get_model_info():
    return {
        "model_name": "SimpleRNN Sentiment Classifier",
        "framework": "PyTorch",
        "dataset": "IMDB Movie Reviews (50,000 samples)",
        "input_vectorizer": "TF-IDF (max_features=5000)",
        "architecture": {
            "input_features": engine.metadata.get("input_size", 5000),
            "hidden_units": engine.metadata.get("hidden_size", 128),
            "rnn_layers": 1,
            "output_layer": "Linear(128 -> 1) + Sigmoid",
            "loss_function": "BCELoss",
            "optimizer": "Adam"
        },
        "accuracy": engine.metadata.get("test_accuracy", 82.58),
        "status": "Ready" if engine.model is not None else "Fallback Heuristic Mode (Model Artifact Not Trained Yet)"
    }

@app.post("/api/predict")
def predict_sentiment(req: ReviewRequest):
    if not req.text or not req.text.strip():
        raise HTTPException(status_code=400, detail="Review text cannot be empty.")
    return engine.predict(req.text)

@app.post("/api/batch-predict")
def batch_predict_sentiment(req: BatchReviewRequest):
    if not req.reviews:
        raise HTTPException(status_code=400, detail="Reviews list cannot be empty.")
    return engine.batch_predict(req.reviews)

# Mount Static directory
static_dir = os.path.join(os.path.dirname(__file__), "static")
if not os.path.exists(static_dir):
    os.makedirs(static_dir, exist_ok=True)

app.mount("/static", StaticFiles(directory=static_dir), name="static")

@app.get("/")
def serve_index():
    index_path = os.path.join(static_dir, "index.html")
    if os.path.exists(index_path):
        return FileResponse(index_path)
    return {"message": "Frontend static files loading..."}

if __name__ == "__main__":
    print("Starting FastAPI Server for RNN Sentiment Analysis on http://127.0.0.1:8000 ...", flush=True)
    uvicorn.run(app, host="127.0.0.1", port=8000)
