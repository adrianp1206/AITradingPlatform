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

def backtest_xgboost(ticker, 
                     start_date='2022-01-01', 
                     end_date='2024-01-01', 
                     feature_subset=None,
                     capital=10000):
    """
    Backtest a saved XGBoost model on unseen data (one ticker at a time),
    allocating `capital` dollars to the stock whenever the model predicts "Up"
    for the next day.

    We still shift the target by 1 day (predict tomorrow's movement using today's data),
    but now the daily P/L is in actual dollars rather than 'points per share'.

    Returns a dictionary of metrics and also the per-day P/L (so you can combine
    multiple tickers' P/L outside this function).
    """
    # ---------------------------
    # 1. Load the XGBoost model
    # ---------------------------
    model_filename = f"../models/boost/xgboost_{ticker}_new.joblib"
    model = load(model_filename)
    print(f"Loaded model from {model_filename}")

    # ---------------------------------------------------
    # 2. Fetch data and calculate technical indicators
    # ---------------------------------------------------
    data = fetch_stock_data_alpha(ticker, start_date=start_date, end_date=end_date)
    data = calculate_technical_indicators(data)

    # ----------------------------------------------------------------------------
    # 3. SHIFT the target to predict tomorrow's movement:
    #    Create 'Future_Close' = 'Close'.shift(-1)
    #    Price_Change = Future_Close - Close
    #    Target = 1 if Price_Change > 0 else 0
    # ----------------------------------------------------------------------------
    data['Future_Close'] = data['Close'].shift(-1)
    data['Price_Change'] = data['Future_Close'] - data['Close']
    data['Target'] = (data['Price_Change'] > 0).astype(int)
    data.dropna(subset=['Future_Close'], inplace=True)

    # -------------------------------------------------------
    # 4. Feature Selection: Ensure consistency with training
    # -------------------------------------------------------
    if feature_subset is None:
        X = data.drop(columns=['Price_Change', 'Target', 'Close'], errors='ignore')
    else:
        X = data[feature_subset]
    y = data['Target']

    # -----------------------------------------------------
    # 5. Prepare X_test, y_test and close_prices
    # -----------------------------------------------------
    if feature_subset is None:
        X_test = data.drop(columns=['Price_Change', 'Target', 'Close'], errors='ignore')
    else:
        X_test = data[feature_subset]
    y_test = data['Target']
    close_prices = data['Close']  # used for computing P/L

    # -------------------------------------------------------------
    # 6. Predict with the model (labels and probabilities)
    # -------------------------------------------------------------
    y_pred = model.predict(X_test)
    y_pred_proba = model.predict_proba(X_test)[:, 1]

    # ----------------------------
    # 7. Calculate classification metrics
    # ----------------------------
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, zero_division=0)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    roc_auc = roc_auc_score(y_test, y_pred_proba)

    # Confusion matrix (comment out if you don't want the plot)
    cm = confusion_matrix(y_test, y_pred)
    ConfusionMatrixDisplay(cm, display_labels=["Down", "Up"]).plot(cmap="Blues")
    plt.title(f"Confusion Matrix for {ticker}")
    plt.show()

    # ------------------------------------------------------------
    # 8. Calculate daily P/L with SHIFTED positions and $10k capital
    # ------------------------------------------------------------
    # Model's raw signals: 1 if we predict "Up" tomorrow, else 0.
    positions = pd.Series(y_pred, index=data.index)
    # SHIFT by 1 to represent that we only know the signal at the end of day t-1
    positions_shifted = positions.shift(1, fill_value=0)

    # Daily price changes
    daily_price_changes = close_prices.diff().fillna(0)

    # How many shares if we invest `capital` at the PREVIOUS day's close
    # shares[today] = positions[t-1] * (capital / close[t-1])
    # (we fill missing close[t-1] with something, or skip the first day)
    prev_close = close_prices.shift(1)
    shares_held = positions_shifted * (capital / prev_close)
    shares_held = shares_held.fillna(0)

    # daily_pnl = shares_held[today] * (close[today] - close[t-1])
    daily_pnl = shares_held * daily_price_changes
    cumulative_pnl = daily_pnl.cumsum()

    total_pnl_value = 0.0
    avg_daily_pnl = 0.0
    max_drawdown = 0.0
    if not cumulative_pnl.empty:
        total_pnl_value = cumulative_pnl.iloc[-1]
        avg_daily_pnl = daily_pnl.mean()
        max_drawdown = (cumulative_pnl - np.maximum.accumulate(cumulative_pnl)).min()

    # ------------------------------------------------------------
    # 9. Compute Additional Performance Metrics: % Gain, Annualized Return
    # ------------------------------------------------------------
    # For *this ticker*, your "investment" is always `capital`.
    # So percent_gain = (final_pnl / capital) * 100
    percent_gain = (total_pnl_value / capital) * 100 if capital != 0 else 0

    # Annualized percent gain (CAGR) over the date range
    start_dt = pd.to_datetime(start_date)
    end_dt = pd.to_datetime(end_date)
    T_years = (end_dt - start_dt).days / 365.25
    if T_years <= 0:
        T_years = 1.0  # fallback to avoid division by zero

    annualized_gain = 0.0
    if percent_gain > -100:  # to avoid negative or zero base in CAGR
        annualized_gain = ((1 + percent_gain / 100) ** (1 / T_years) - 1) * 100

    print(f"Total P/L: ${total_pnl_value:.2f}")
    print(f"Percent Gain: {percent_gain:.2f}%")
    print(f"Annualized Percent Gain: {annualized_gain:.2f}%")

    # ------------------------------------------------------------
    # 10. Plot the cumulative P&L over time (optional)
    # ------------------------------------------------------------
    plt.figure(figsize=(12, 6))
    plt.plot(cumulative_pnl, label='Cumulative P/L', color='green')
    plt.title(f"Cumulative P/L for {ticker} Backtest (Shifted Target, $10k)")
    plt.xlabel('Days')
    plt.ylabel('Profit/Loss ($)')
    plt.legend()
    plt.show()

    # ---------------------------
    # Return a dictionary of metrics + daily_pnl
    # ---------------------------
    metrics = {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1_score': f1,
        'roc_auc': roc_auc,
        'total_pnl': total_pnl_value,
        'avg_daily_pnl': avg_daily_pnl,
        'max_drawdown': max_drawdown,
        'percent_gain': percent_gain,
        'annualized_gain': annualized_gain,
        'daily_pnl': daily_pnl  # so we can sum across tickers for a portfolio
    }

    return metrics
