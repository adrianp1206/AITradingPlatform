import sys
import os

# Add the parent directory (ML/) to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import pandas as pd
import numpy as np
import xgboost as xgb
from joblib import load
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, roc_auc_score,
    confusion_matrix, ConfusionMatrixDisplay
)
import matplotlib.pyplot as plt
from data_processing import fetch_stock_data_alpha, calculate_technical_indicators

def backtest_xgboost(ticker, start_date='2022-01-01', end_date='2024-01-01', feature_subset=None):
    """
    Backtest a saved XGBoost model on unseen data, with the target shifted by 1 day.
    We predict tomorrow's movement using today's data.
    """

    # ---------------------------
    # 1. Load the XGBoost model
    # ---------------------------
    model_filename = f"../models/boost/xgboost_{ticker}.joblib"
    model = load(model_filename)
    print(f"Loaded model from {model_filename}")

    # ---------------------------------------------------
    # 2. Fetch data and calculate technical indicators
    # ---------------------------------------------------
    data = fetch_stock_data_alpha(ticker, start_date=start_date, end_date=end_date)
    data = calculate_technical_indicators(data)

    # ----------------------------------------------------------------------------
    # 3. SHIFT the target to predict tomorrow's movement
    #    * We create 'Future_Close' = 'Close'.shift(-1)
    #    * Price_Change = Future_Close - Close
    #    * Target = 1 if Price_Change > 0 else 0
    # ----------------------------------------------------------------------------
    data['Future_Close'] = data['Close'].shift(-1)
    data['Price_Change'] = data['Future_Close'] - data['Close']
    data['Target'] = (data['Price_Change'] > 0).astype(int)

    # Drop rows where 'Future_Close' is NaN (the last row usually)
    data.dropna(subset=['Future_Close'], inplace=True)

    # Keep relevant features + 'Price_Change', 'Target', 'Close'
    if feature_subset is None:
        # If no subset provided, use all columns except these
        relevant_columns = list(data.columns)
        # remove duplicates if they exist
        relevant_columns = list(dict.fromkeys(relevant_columns))
    else:
        relevant_columns = feature_subset + ['Price_Change', 'Target', 'Close']

    data = data[relevant_columns].dropna()  # ensure we drop any remaining NaNs

    # Check shape and columns
    print("Data columns after shifting target:", data.columns.tolist())
    print("Data shape after shifting target and dropna:", data.shape)

    # ----------------------------------------------------
    # 4. Prepare X_test and y_test for the backtest period
    # ----------------------------------------------------
    if feature_subset is None:
        X_test = data.drop(columns=['Price_Change', 'Target', 'Close'], errors='ignore')
    else:
        X_test = data[feature_subset]
    y_test = data['Target']
    close_prices = data['Close']  # for P/L calculation

    # -------------------------------------------------------------
    # 5. Predict with the model (labels and probabilities)
    # -------------------------------------------------------------
    y_pred = model.predict(X_test)
    y_pred_proba = model.predict_proba(X_test)[:, 1]

    # ----------------------------
    # 6. Calculate classification metrics
    # ----------------------------
    metrics = {
        'accuracy': accuracy_score(y_test, y_pred),
        'precision': precision_score(y_test, y_pred, zero_division=0),
        'recall': recall_score(y_test, y_pred),
        'f1_score': f1_score(y_test, y_pred),
        'roc_auc': roc_auc_score(y_test, y_pred_proba)
    }

    # Confusion Matrix
    cm = confusion_matrix(y_test, y_pred)
    ConfusionMatrixDisplay(cm, display_labels=["Down", "Up"]).plot(cmap="Blues")
    plt.title(f"Confusion Matrix for {ticker}")
    plt.show()

    # ------------------------------------------------------------
    # 7. Calculate P/L with SHIFTED positions
    #    positions[i] = 1 if model says "tomorrow up", else 0
    #    But the day i's returns come from day i+1's price move,
    #    so we SHIFT positions by 1 day to reflect that you only
    #    know today's prediction for tomorrow at the end of today.
    # ------------------------------------------------------------
    positions = pd.Series(y_pred, index=data.index)  # day-i prediction
    price_changes = close_prices.diff().fillna(0)    # day-to-day price changes

    # SHIFT positions by 1 day so you hold the position
    # *during* the price change from day i to day i+1
    positions_shifted = positions.shift(1, fill_value=0)

    # Daily returns = positions (previous day's signal) * today's price change
    daily_returns = positions_shifted * price_changes

    # Cumulative P&L
    cumulative_pnl = np.cumsum(daily_returns)
    metrics['total_pnl'] = cumulative_pnl.iloc[-1]
    metrics['avg_daily_pnl'] = daily_returns.mean()
    metrics['max_drawdown'] = (cumulative_pnl - np.maximum.accumulate(cumulative_pnl)).min()

    # ----------------------------------
    # 8. Plot the cumulative P&L
    # ----------------------------------
    plt.figure(figsize=(12, 6))
    plt.plot(cumulative_pnl, label='Cumulative P/L', color='green')
    plt.title(f"Cumulative P/L for {ticker} Backtest (Shifted Target)")
    plt.xlabel('Days')
    plt.ylabel('Profit/Loss ($)')
    plt.legend()
    plt.show()

    return metrics
