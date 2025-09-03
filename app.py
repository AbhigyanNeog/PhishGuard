#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import re
import math
import joblib
import pandas as pd
from urllib.parse import urlparse
from flask import Flask, render_template_string, request

# ---------------------------
# Feature Helpers (same as train_model.py)
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
# Flask App
# ---------------------------

app = Flask(__name__)

# Load trained model
model = joblib.load("model/phish_model.pkl")

# Stylish HTML template (Bootstrap + modern card UI)
TEMPLATE = """
<!doctype html>
<html lang="en">
  <head>
    <meta charset="utf-8">
    <meta name="viewport" content="width=device-width, initial-scale=1">
    <title>PhishGuard</title>
    <link href="https://cdn.jsdelivr.net/npm/bootstrap@5.3.2/dist/css/bootstrap.min.css" rel="stylesheet">
    <style>
      body {
        background: linear-gradient(135deg, #74ebd5 0%, #ACB6E5 100%);
        height: 100vh;
        display: flex;
        align-items: center;
        justify-content: center;
        font-family: Arial, sans-serif;
      }
      .card {
        border-radius: 20px;
        box-shadow: 0 8px 20px rgba(0,0,0,0.2);
      }
      .btn-custom {
        background-color: #007bff;
        color: white;
        border-radius: 30px;
        padding: 10px 20px;
      }
      .result-safe {
        color: green;
        font-weight: bold;
      }
      .result-phish {
        color: red;
        font-weight: bold;
      }
    </style>
  </head>
  <body>
    <div class="container">
      <div class="card p-5 text-center">
        <h2 class="mb-4">🔒 PhishGuard</h2>
        <p class="text-muted">Enter a URL below to check if it is Safe or a Phishing attempt</p>
        <form method="post" class="d-flex justify-content-center mb-3">
          <input type="text" name="url" class="form-control me-2" placeholder="https://example.com" required style="max-width: 500px;">
          <button type="submit" class="btn btn-custom">Check</button>
        </form>
        {% if result is not none %}
          <h4 class="{{ 'result-safe' if 'SAFE' in result else 'result-phish' }}">Result: {{ result }}</h4>
        {% endif %}
      </div>
    </div>
  </body>
</html>
"""

@app.route("/", methods=["GET", "POST"])
def index():
    result = None
    if request.method == "POST":
        url = request.form["url"]
        X = dataframe_from_urls(pd.Series([url]))
        prediction = model.predict(X.values)[0]
        result = "⚠️ PHISHING" if prediction == 1 else "✅ SAFE"
    return render_template_string(TEMPLATE, result=result)

if __name__ == "__main__":
    app.run(debug=True)




