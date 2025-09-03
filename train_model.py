#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Train a phishing URL classifier with 10 features and save it to model/phish_model.pkl
"""

import os
import re
import math
import joblib
import pandas as pd
from urllib.parse import urlparse
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
from sklearn.ensemble import RandomForestClassifier

# ---------------------------
# Feature Helpers
# ---------------------------

def shannon_entropy(s: str) -> float:
    if not s:
        return 0.0
    probs = [float(s.count(c)) / len(s) for c in set(s)]
    return -sum(p * math.log(p, 2) for p in probs)

def is_ipv4(host: str) -> bool:
    if not host:
        return False
    m = re.match(r"^\d{1,3}(?:\.\d{1,3}){3}$", host.strip())
    if not m:
        return False
    parts = host.split(".")
    return all(0 <= int(p) <= 255 for p in parts)

def extract_features_from_url(url: str) -> dict:
    parsed = urlparse(url.strip())
    host = parsed.netloc or ""
    path = parsed.path or ""

    return {
        "url_len": len(url),
        "host_len": len(host),
        "path_len": len(path),
        "num_dots": url.count("."),
        "num_hyphens": url.count("-"),
        "num_at": url.count("@"),
        "num_digits": sum(ch.isdigit() for ch in url),
        "uses_https": 1 if parsed.scheme.lower() == "https" else 0,
        "has_ip_host": 1 if is_ipv4(host) else 0,
        "entropy_url": shannon_entropy(url),
    }

def dataframe_from_urls(urls: pd.Series) -> pd.DataFrame:
    rows = [extract_features_from_url(u) for u in urls.fillna("")]
    return pd.DataFrame(rows)

# ---------------------------
# Training Pipeline
# ---------------------------

def main():
    # Paths
    data_path = os.path.join("dataset", "phishing_dataset.csv")
    out_dir = os.path.join("model")
    out_model = os.path.join(out_dir, "phish_model.pkl")

    if not os.path.exists(data_path):
        raise FileNotFoundError(
            f"Dataset not found at {data_path}. "
            "Create dataset/phishing_dataset.csv with columns: url,label (1=phish, 0=safe)"
        )

    os.makedirs(out_dir, exist_ok=True)

    # Load dataset
    df = pd.read_csv(data_path)
    if "url" not in df.columns or "label" not in df.columns:
        raise ValueError("CSV must contain columns: url,label")

    df = df.dropna(subset=["url", "label"]).reset_index(drop=True)

    # Features + labels
    X = dataframe_from_urls(df["url"])
    y = df["label"].astype(int).values

    # Split
    X_train, X_test, y_train, y_test = train_test_split(
        X.values, y, test_size=0.2, random_state=42, stratify=y
    )

    # Model
    clf = RandomForestClassifier(
        n_estimators=200,
        random_state=42,
        class_weight="balanced",
        n_jobs=-1
    )
    clf.fit(X_train, y_train)

    # Eval
    y_pred = clf.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    print("\n=== Evaluation ===")
    print(f"Accuracy: {acc:.4f}")
    print("\nClassification report:\n", classification_report(y_test, y_pred, digits=4))

    # Save model
    joblib.dump(clf, out_model)
    print(f"\nModel saved to: {out_model}")

    # Quick sanity check
    samples = [
        "https://openai.com/",
        "http://paypal.com-security-login.com",
        "http://198.51.100.42/login",
        "https://google.com",
    ]
    print("\n=== Quick sanity check ===")
    Xs = dataframe_from_urls(pd.Series(samples))
    preds = clf.predict(Xs.values)
    for u, p in zip(samples, preds):
        print(f"{'PHISH' if p==1 else 'SAFE '}  -  {u}")

if __name__ == "__main__":
    main()
