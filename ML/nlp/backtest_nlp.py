import sys
import os

# Add the parent directory (ML/) to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

from data_processing import fetch_stock_data_alpha
from run_nlp import generate_next_day_weighted_rolling_sentiments

def get_predicted_signal(score):
    return "buy" if score > 0 else "sell"

def get_actual_signal(row):
    return "buy" if row['Close'] > row['Open'] else "sell"

def backtest_nlp_signals(ticker, start_date, end_date, api_key, window=15):
    """
    Backtest NLP sentiment-based trading strategy using real price data.
    """
    # 1. Fetch sentiment data
    sentiment_df = generate_next_day_weighted_rolling_sentiments(ticker, start_date, end_date, window=window)
    if sentiment_df is None or sentiment_df.empty:
        raise ValueError("No sentiment data returned.")

    # 2. Fetch stock price data
    stock_df = fetch_stock_data_alpha(ticker, api_key=api_key, start_date=start_date, end_date=end_date)
    stock_df = stock_df[['Close', 'Open']].copy()
    stock_df['Date'] = stock_df.index.normalize()

    # 3. Merge on Date
    merged_df = pd.merge(sentiment_df, stock_df, on='Date', how='inner')

    if merged_df.empty:
        raise ValueError("Merged DataFrame is empty. Check date ranges or ticker symbol.")

    # 4. Compute signals
    merged_df['predicted_signal'] = merged_df['next_day_sentiment'].apply(get_predicted_signal)
    merged_df['actual_signal'] = merged_df.apply(get_actual_signal, axis=1)

    # 5. Evaluate performance
    labels = ['buy', 'sell']
    cm = confusion_matrix(merged_df['actual_signal'], merged_df['predicted_signal'], labels=labels)
    acc = accuracy_score(merged_df['actual_signal'], merged_df['predicted_signal'])

    print("Confusion Matrix:")
    print(cm)
    print(f"Accuracy: {acc:.2f}")

    # 6. Plot confusion matrix
    plt.figure(figsize=(6, 4))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=labels, yticklabels=labels)
    plt.title(f"Confusion Matrix for {ticker} NLP Signals - Weighted ({window}-Day Window)")
    plt.ylabel("Actual Signal")
    plt.xlabel("Predicted Signal")
    plt.tight_layout()
    plt.show()

    return merged_df, cm, acc