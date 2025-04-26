#!/usr/bin/env python3
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import pearsonr
from datetime import datetime, timedelta

import firebase_admin
from firebase_admin import credentials, firestore

from data_processing import fetch_stock_data_alpha
from run_all_model import (
    get_next_trading_day,
    predict_lstm,
    predict_xgboost,
    predict_nlp,
    predict_rl
)

# ──────────────────────────────────────────────────────────────────────────────
# Initialize Firestore
# ──────────────────────────────────────────────────────────────────────────────
try:
    firebase_admin.get_app()
except ValueError:
    here = os.path.dirname(__file__)
    # go up one level, then firebase.json
    cred_path = os.path.abspath(os.path.join(here, "..", "firebase.json"))
    if not os.path.isfile(cred_path):
        raise FileNotFoundError(f"Could not find firebase.json at {cred_path}")
    cred = credentials.Certificate(cred_path)
    firebase_admin.initialize_app(cred)
db = firestore.client()

def fetch_predictions_from_db(tickers=None):
    """
    Pulls every doc in StockSignals/<ticker>/predictions
    into one DataFrame. If tickers is None, it lists
    all tickers under StockSignals.
    """
    if tickers is None:
        tickers = [doc.id for doc in db.collection("StockSignals").stream()]

    rows = []
    for ticker in tickers:
        coll = db.collection("StockSignals").document(ticker).collection("predictions")
        for doc in coll.stream():
            d = doc.to_dict()
            d["ticker"] = ticker
            rows.append(d)

    df = pd.DataFrame(rows)
    df["next_trading_date"] = pd.to_datetime(df["next_trading_date"])
    df["updated_at"] = pd.to_datetime(df["updated_at"])
    return df

def analyze_predictions_df(preds, api_key, actual_start_date, actual_end_date):
    """
    preds: DataFrame with columns
      ticker, lstm_predicted_price, next_trading_date,
      xgboost_signal, xgboost_prob, nlp_sentiment_score,
      rl_recommendation, updated_at
    """
    preds = preds.copy()
    preds["next_trading_date"] = pd.to_datetime(preds["next_trading_date"])

    comparisons = []
    for ticker in preds["ticker"].unique():
        print(f"Processing ticker: {ticker}")
        pred_ticker = preds[preds["ticker"] == ticker]

        # Fetch actual historical data
        actual_df = fetch_stock_data_alpha(
            ticker,
            api_key=api_key,
            start_date=actual_start_date,
            end_date=actual_end_date
        )
        # Ensure a Date column
        if "Date" not in actual_df.columns:
            if isinstance(actual_df.index, pd.DatetimeIndex):
                actual_df = actual_df.reset_index().rename(columns={"index":"Date"})
            else:
                days = pd.date_range(start=actual_start_date, end=actual_end_date, freq="B")
                if len(days) == len(actual_df):
                    actual_df["Date"] = days
                else:
                    raise ValueError("Row count mismatch for business days")

        actual_df["Date"] = pd.to_datetime(actual_df["Date"])
        actual_df = actual_df.sort_values("Date")

        # Compare each prediction to the real close
        for _, row in pred_ticker.iterrows():
            pred_date = row["next_trading_date"]
            prev = actual_df[actual_df["Date"] < pred_date]
            if prev.empty:
                continue
            prev_close = prev.iloc[-1]["Close"]

            today = actual_df[actual_df["Date"] == pred_date]
            if today.empty:
                continue
            actual_close = today.iloc[0]["Close"]

            # compute return
            pct_return = (actual_close - prev_close) / prev_close

            # derive signals
            lstm_signal = "Up" if row["lstm_predicted_price"] > prev_close else "Down"
            xgb_signal  = row["xgboost_signal"]
            rl       = row["rl_recommendation"].strip().lower()
            rl_signal = "Up" if rl=="buy" else ("Down" if rl=="sell" else "Hold")

            def profit(sig):
                if sig=="Up":   return actual_close - prev_close
                if sig=="Down": return prev_close - actual_close
                return 0.0

            comparisons.append({
                "ticker": ticker,
                "next_trading_date": pred_date,
                "prev_close": prev_close,
                "actual_close": actual_close,
                "actual_return": pct_return,
                "lstm_predicted_price": row["lstm_predicted_price"],
                "lstm_signal": lstm_signal,
                "lstm_profit": profit(lstm_signal),
                "xgboost_prob": row["xgboost_prob"],
                "xgboost_signal": xgb_signal,
                "xgboost_profit": profit(xgb_signal),
                "rl_recommendation": row["rl_recommendation"],
                "rl_signal": rl_signal,
                "rl_profit": profit(rl_signal),
                "nlp_sentiment_score": row["nlp_sentiment_score"],
                "updated_at": row["updated_at"]
            })

    comp_df = pd.DataFrame(comparisons)
    comp_df["lstm_pct"]    = comp_df["lstm_profit"]    / comp_df["prev_close"] * 100
    comp_df["xgboost_pct"] = comp_df["xgboost_profit"] / comp_df["prev_close"] * 100
    comp_df["rl_pct"]      = comp_df["rl_profit"]      / comp_df["prev_close"] * 100

    # … insert your MAE/RMSE/etc calculations and plotting here …

    return comp_df

if __name__ == "__main__":
    API_KEY = "NL9PDOM5JWRPAT9O"
    START   = "2025-04-02"
    END     = "2025-04-14"

    # 1) fetch
    df_preds = fetch_predictions_from_db(tickers=["TSLA", "MSFT", "KO"])
    # 2) analyze
    comp_df  = analyze_predictions_df(
        df_preds,
        api_key=API_KEY,
        actual_start_date=START,
        actual_end_date=END
    )

    # e.g. save results or just print
    comp_df.to_csv("comparison_results.csv", index=False)
    print(comp_df)
