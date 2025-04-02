import sys
import os

# Add the parent directory (ML/) to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import pandas as pd
import numpy as np
import xgboost as xgb
from joblib import load
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt
from data_processing import fetch_stock_data_alpha, calculate_technical_indicators

# Backtesting function
def backtest_xgboost(ticker, start_date='2022-01-01', end_date='2024-01-01', feature_subset=None):
    """
    Backtest a saved XGBoost model on unseen data.

    Args:
    ticker: str, The stock ticker symbol.
    start_date: str, Start date for backtesting data.
    end_date: str, End date for backtesting data.
    feature_subset: list of str, Features used during model training.

    Returns:
    metrics: dict, Performance metrics for the backtested data.
    """
    # Load the saved model
    model_filename = f"../models/boost/xgboost_{ticker}.joblib"
    model = load(model_filename)
    print(f"Loaded model from {model_filename}")
    
    # Fetch and preprocess backtesting data
    data = fetch_stock_data_alpha(ticker, start_date=start_date, end_date=end_date)
    data = calculate_technical_indicators(data)
    data['Price_Change'] = data['Close'].diff()
    data['Target'] = (data['Price_Change'] > 0).astype(int)

    # Only keep relevant features + target
    relevant_columns = feature_subset + ['Price_Change', 'Target', 'Close']
    data = data[relevant_columns]

    # Log missingness
    missing = data.isnull().sum()
    print("Missing values before dropna:\n", missing[missing > 0])

    # Drop rows with NaNs in the relevant columns
    data = data.dropna()

    if data.empty:
        raise ValueError(f"After dropping NaNs, no data is left for {ticker}.")

    print("Available columns:", data.columns.tolist())
    print("Data shape before dropna:", data.shape)
    print("Data shape after dropna:", data.dropna().shape)


    X_test = data[feature_subset] if feature_subset else data.drop(columns=['Price_Change', 'Target', 'Close'])
    y_test = data['Target']
    close_prices = data['Close']

    print("Available columns:", data.columns.tolist())
    print("Data shape before dropna:", data.shape)
    print("Data shape after dropna:", data.dropna().shape)

    
    # Predict on the backtesting data
    y_pred = model.predict(X_test)
    y_pred_proba = model.predict_proba(X_test)[:, 1]
    
    # Calculate metrics
    metrics = {
        'accuracy': accuracy_score(y_test, y_pred),
        'precision': precision_score(y_test, y_pred, zero_division=0),
        'recall': recall_score(y_test, y_pred),
        'f1_score': f1_score(y_test, y_pred),
        'roc_auc': roc_auc_score(y_test, y_pred_proba)
    }
    
    # Display confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    ConfusionMatrixDisplay(cm, display_labels=["Down", "Up"]).plot(cmap="Blues")
    plt.title(f"Confusion Matrix for {ticker}")
    plt.show()

    positions = y_pred  # 1 for "up" (buy), 0 for "down" (sell)
    price_changes = close_prices.diff().fillna(0)  # Daily price changes
    daily_returns = positions * price_changes  # Profit/loss based on predictions
    cumulative_pnl = np.cumsum(daily_returns)  # Cumulative P/L
    
    # Add P/L to metrics
    metrics['total_pnl'] = cumulative_pnl.iloc[-1]
    metrics['avg_daily_pnl'] = daily_returns.mean()
    metrics['max_drawdown'] = np.min(cumulative_pnl - np.maximum.accumulate(cumulative_pnl))
    
    # Plot cumulative P/L
    plt.figure(figsize=(12, 6))
    plt.plot(cumulative_pnl, label='Cumulative P/L', color='green')
    plt.title(f"Cumulative P/L for {ticker} Backtest")
    plt.xlabel('Days')
    plt.ylabel('Profit/Loss ($)')
    plt.legend()
    plt.show()
    
    return metrics