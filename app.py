"""FastAPI app: REST API + web UI for spam detection.

Run:  uvicorn app:app --reload
Open: http://127.0.0.1:8000        (web UI)
      http://127.0.0.1:8000/docs   (interactive API docs)
"""
import re

import joblib
import numpy as np
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from preprocess import clean_text

MODEL_PATH = "models/spam_model.joblib"

app = FastAPI(title="Spam Detector API", version="1.0")

try:
    pipeline = joblib.load(MODEL_PATH)
except FileNotFoundError:
    pipeline = None

tfidf = pipeline.named_steps["tfidf"] if pipeline else None
clf = pipeline.named_steps["clf"] if pipeline else None


class Message(BaseModel):
    text: str = Field(..., min_length=1, max_length=20000)


def spam_weights() -> np.ndarray:
    """Per-feature weight toward the spam class (works for NB and LR)."""
    if hasattr(clf, "coef_"):
        return clf.coef_[0]
    return clf.feature_log_prob_[1] - clf.feature_log_prob_[0]


def spam_signals(text: str, top_k: int = 8):
    """Original words that pushed the prediction toward spam."""
    vec = tfidf.transform([clean_text(text)])
    weights = spam_weights()
    contrib = {}
    for idx, val in zip(vec.indices, vec.data):
        contrib[tfidf.get_feature_names_out()[idx]] = val * weights[idx]
    seen, signals = set(), []
    for word in re.findall(r"[A-Za-z']+", text):
        stem = clean_text(word)
        score = contrib.get(stem, 0)
        if score > 0 and word.lower() not in seen:
            seen.add(word.lower())
            signals.append((word, score))
    signals.sort(key=lambda x: -x[1])
    return [w for w, _ in signals[:top_k]]


@app.get("/health")
def health():
    return {"status": "ok", "model_loaded": pipeline is not None}


@app.post("/predict")
def predict(msg: Message):
    if pipeline is None:
        raise HTTPException(503, "Model not trained yet. Run: python train.py")
    proba = float(pipeline.predict_proba([clean_text(msg.text)])[0][1])
    return {
        "label": "spam" if proba >= 0.5 else "ham",
        "spam_probability": round(proba, 4),
        "spam_signals": spam_signals(msg.text),
    }


app.mount("/static", StaticFiles(directory="static"), name="static")


@app.get("/")
def index():
    return FileResponse("static/index.html")
