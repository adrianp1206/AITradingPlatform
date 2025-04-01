import sys
import os

# Add the parent directory (ML/) to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model
from sklearn.metrics import mean_absolute_error, mean_squared_error, mean_absolute_percentage_error, r2_score
from lstm_model import Attention
from data_processing import fetch_stock_data_alpha, preprocess_data_alpha, create_lstm_input


def backtest_lstm(ticker, model_path, api_key, start_date='2022-01-01', end_date='2024-01-01'):
    # Fetch and preprocess data
    data = fetch_stock_data_alpha(ticker, api_key=api_key, start_date=start_date, end_date=end_date)
    data, scaler = preprocess_data_alpha(data)

    # Prepare input data for backtesting
    X, y = create_lstm_input(data, target_column='Close', lookback=20)

    # Load the saved LSTM model
    model = load_model(model_path, custom_objects={'Attention': Attention})

    # Generate predictions
    y_pred = model.predict(X)

    # Inverse transform predictions and actual values to original scale
    y_test_padded = np.zeros((len(y), scaler.min_.shape[0]))
    y_pred_padded = np.zeros((len(y_pred), scaler.min_.shape[0]))
    y_test_padded[:, 0] = y.flatten()
    y_pred_padded[:, 0] = y_pred.flatten()

    y_actual = scaler.inverse_transform(y_test_padded)[:, 0]
    y_predicted = scaler.inverse_transform(y_pred_padded)[:, 0]

    # Calculate metrics
    mae = mean_absolute_error(y_actual, y_predicted)
    rmse = np.sqrt(mean_squared_error(y_actual, y_predicted))
    mape = mean_absolute_percentage_error(y_actual, y_predicted)
    r2 = r2_score(y_actual, y_predicted)

    print(f"Backtesting Results for {ticker} ({start_date} to {end_date}):")
    print(f"MAE: {mae:.2f}, RMSE: {rmse:.2f}, MAPE: {mape * 100:.2f}%, R²: {r2:.4f}")

    # Plot actual vs predicted prices
    plt.figure(figsize=(12, 6))
    plt.plot(y_actual, label='Actual Prices', color='blue', marker='o')
    plt.plot(y_predicted, label='Predicted Prices', color='red', marker='x')
    plt.title(f'Backtesting: Actual vs Predicted Prices for {ticker}')
    plt.xlabel('Days')
    plt.ylabel('Price')
    plt.legend()
    plt.show()

    return mae, rmse, mape, r2