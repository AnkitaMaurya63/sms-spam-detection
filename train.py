"""Train and compare Naive Bayes and Logistic Regression for SMS spam detection.

Usage:
    python train.py                      # auto-loads / downloads the dataset
    python train.py --data path/to/file  # use your own copy
"""
import argparse
import io
import json
import os
import urllib.request
import zipfile

import joblib
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, classification_report,
                             confusion_matrix, f1_score, precision_score,
                             recall_score)
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline

from preprocess import clean_text

DATA_URL = "https://archive.ics.uci.edu/static/public/228/sms+spam+collection.zip"
DEFAULT_PATHS = ["data/SMSSpamCollection", "data/spam.csv"]
RANDOM_STATE = 42


def load_data(path=None) -> pd.DataFrame:
    """Return a DataFrame with columns: label (ham/spam), text."""
    if path is None:
        path = next((p for p in DEFAULT_PATHS if os.path.exists(p)), None)

    if path is None:
        print("Dataset not found locally, downloading from UCI ...")
        raw = urllib.request.urlopen(DATA_URL, timeout=60).read()
        with zipfile.ZipFile(io.BytesIO(raw)) as z:
            z.extractall("data")
        path = "data/SMSSpamCollection"

    if path.endswith(".csv"):  # Kaggle format: v1 = label, v2 = text
        df = pd.read_csv(path, encoding="latin-1")[["v1", "v2"]]
        df.columns = ["label", "text"]
    else:  # UCI format: label <TAB> text
        df = pd.read_csv(path, sep="\t", header=None, names=["label", "text"],
                         encoding="latin-1")
    df = df.dropna().drop_duplicates().reset_index(drop=True)
    print(f"Loaded {len(df)} messages from {path}")
    return df


def build_pipeline(model) -> Pipeline:
    return Pipeline([
        ("tfidf", TfidfVectorizer(ngram_range=(1, 2), min_df=2, sublinear_tf=True)),
        ("clf", model),
    ])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default=None, help="path to dataset file")
    args = parser.parse_args()

    df = load_data(args.data)
    print(df["label"].value_counts().to_string(), "\n")

    df["clean"] = df["text"].apply(clean_text)
    y = (df["label"] == "spam").astype(int)

    X_train, X_test, y_train, y_test = train_test_split(
        df["clean"], y, test_size=0.2, stratify=y, random_state=RANDOM_STATE)

    models = {
        "Naive Bayes": MultinomialNB(alpha=0.1),
        "Logistic Regression": LogisticRegression(
            max_iter=1000, class_weight="balanced", random_state=RANDOM_STATE),
    }

    results, fitted = {}, {}
    for name, model in models.items():
        pipe = build_pipeline(model).fit(X_train, y_train)
        pred = pipe.predict(X_test)
        results[name] = {
            "accuracy": round(accuracy_score(y_test, pred), 4),
            "precision": round(precision_score(y_test, pred), 4),
            "recall": round(recall_score(y_test, pred), 4),
            "f1": round(f1_score(y_test, pred), 4),
            "confusion_matrix": confusion_matrix(y_test, pred).tolist(),
        }
        fitted[name] = pipe
        print(f"=== {name} ===")
        print(classification_report(y_test, pred, target_names=["ham", "spam"]))
        print("Confusion matrix [[TN, FP], [FN, TP]]:")
        print(confusion_matrix(y_test, pred), "\n")

    best = max(results, key=lambda n: results[n]["f1"])
    print(f"Best model by spam F1: {best}")

    os.makedirs("models", exist_ok=True)
    os.makedirs("results", exist_ok=True)
    joblib.dump(fitted[best], "models/spam_model.joblib")
    with open("results/metrics.json", "w") as f:
        json.dump({"best_model": best, "test_size": len(y_test),
                   "results": results}, f, indent=2)
    print("Saved models/spam_model.joblib and results/metrics.json")


if __name__ == "__main__":
    main()
