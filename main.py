from fastapi import FastAPI
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware
import yfinance as yf
import pandas as pd
import numpy as np
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

app = FastAPI()

# Allow frontend to call backend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Request schema
class PredictRequest(BaseModel):
    ticker: str
    index: str
    start: str
    end: str
    features: dict
    label: str
    kernel: str = "rbf"

@app.post("/api/predict")
def predict(req: PredictRequest):
    # --- Download stock data ---
    data = yf.download(req.ticker, start=req.start, end=req.end, progress=False)
    if data.empty:
        return {"error": "No data found for ticker."}

    df = data[["Close"]].copy()

    # --- Feature engineering ---
    if req.features.get("stock_volatility", False):
        df["volatility"] = df["Close"].pct_change().rolling(10).std()

    if req.features.get("stock_momentum", False):
        df["momentum"] = df["Close"].pct_change()

    if req.features.get("rsi14", False):
        delta = df["Close"].diff()
        up = delta.clip(lower=0)
        down = -1 * delta.clip(upper=0)
        roll_up = up.rolling(14).mean()
        roll_down = down.rolling(14).mean()
        rs = roll_up / roll_down
        df["rsi14"] = 100 - (100 / (1 + rs))

    df.dropna(inplace=True)

    # --- Label: Next day up (1) or down (0) ---
    df["target"] = (df["Close"].shift(-1) > df["Close"]).astype(int)
    df.dropna(inplace=True)

    X = df.drop(columns=["target"])
    y = df["target"]

    # --- Train/Test Split ---
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)

    # --- Train SVM ---
    model = SVC(kernel=req.kernel, probability=True)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]

    # --- Metrics ---
    metrics = {
        "accuracy": round(accuracy_score(y_test, y_pred), 2),
        "precision": round(precision_score(y_test, y_pred), 2),
        "recall": round(recall_score(y_test, y_pred), 2),
        "f1": round(f1_score(y_test, y_pred), 2),
    }

    return {
        "ticker": req.ticker,
        "start": req.start,
        "end": req.end,
        "dates": X_test.index.strftime("%Y-%m-%d").tolist(),
        "close": df.loc[X_test.index, "Close"].tolist(),
        "y_pred": y_pred.tolist(),
        "y_proba": y_proba.tolist(),
        "metrics": metrics,
    }
