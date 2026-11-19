"""Classify SMS messages with the trained model.

Usage:
    python predict.py "Congratulations! You won a free prize, call now"
    python predict.py            # interactive mode
"""
import sys

import joblib

from preprocess import clean_text

MODEL_PATH = "models/spam_model.joblib"


def classify(model, message: str):
    cleaned = clean_text(message)
    proba = model.predict_proba([cleaned])[0][1]
    return ("SPAM" if proba >= 0.5 else "HAM"), proba


def main():
    model = joblib.load(MODEL_PATH)
    if len(sys.argv) > 1:
        label, p = classify(model, " ".join(sys.argv[1:]))
        print(f"{label} (spam probability: {p:.2%})")
        return
    print("Type a message (empty line to quit):")
    while True:
        msg = input("> ").strip()
        if not msg:
            break
        label, p = classify(model, msg)
        print(f"  {label} (spam probability: {p:.2%})")


if __name__ == "__main__":
    main()
