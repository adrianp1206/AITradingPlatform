import os
import sys
import datetime
import pandas as pd
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from joblib import load

# Import your data processing functions and any other required modules
from data_processing import fetch_stock_data_alpha, calculate_technical_indicators

# -------------------------------
# 1. Helper Functions
# -------------------------------

def get_latest_data(ticker, days=200):
    """
    Pull historical data for the past `days` ending yesterday.
    """
    today = datetime.date.today()
    end_date = today - datetime.timedelta(days=1)  # Assuming yesterday is the last trading day
    start_date = end_date - datetime.timedelta(days=days)
    
    # Replace 'YOUR_API_KEY' with your actual API key
    api_key = 'NL9PDOM5JWRPAT9O'
    data = fetch_stock_data_alpha(ticker, api_key=api_key, start_date=str(start_date), end_date=str(end_date))
    
    # Ensure the 'Date' column exists and is properly formatted
    if 'Date' not in data.columns:
        data['Date'] = pd.to_datetime(data.index).date
    else:
        data['Date'] = pd.to_datetime(data['Date']).dt.date
    
    return data

def get_next_trading_day(last_date):
    """
    Given a date, compute the next trading day by skipping weekends.
    """
    next_day = last_date + datetime.timedelta(days=1)
    # Skip Saturday (5) and Sunday (6)
    while next_day.weekday() >= 5:
        next_day += datetime.timedelta(days=1)
    return next_day

# -------------------------------
# 2. Model Prediction Functions
# -------------------------------

def predict_lstm(ticker, lstm_model_path, data, lookback=20):
    """
    Predict the next day's price using the LSTM model.
    
    Parameters:
    - ticker: str, the stock ticker symbol (used for logging or further customization)
    - lstm_model_path: str, the file path to the saved LSTM model
    - data: pd.DataFrame, historical stock data (must include at least 'Close' and other features)
    - lookback: int, number of days used to create the input sequence (default is 20)
    
    Returns:
    - next_day_price: float, the predicted price for the next trading day
    """
    import sys
    import os
    import numpy as np
    from tensorflow.keras.models import load_model

    # Add the 'lstm' subdirectory (located in ML/lstm) to sys.path so that lstm_model can be imported
    sys.path.append(os.path.join(os.path.dirname(__file__), 'lstm'))

    # Import custom Attention layer and data processing functions
    from lstm_model import Attention
    from data_processing import preprocess_data_alpha

    # Preprocess the data and obtain the scaler (assumes your preprocess function returns both)
    data, scaler = preprocess_data_alpha(data)

    # Ensure there is enough data to form one input sequence
    if len(data) < lookback:
        raise ValueError(f"Not enough data to create an input sequence; need at least {lookback} rows.")

    # Create input sample using the last `lookback` rows from the preprocessed data.
    # We assume that the model was trained using all columns from the processed DataFrame.
    X_next = data.iloc[-lookback:].values  # shape: (lookback, num_features)
    X_next = X_next.reshape((1, lookback, X_next.shape[1]))  # shape: (1, lookback, num_features)

    # Load the LSTM model (using custom Attention layer)
    model = load_model(lstm_model_path, custom_objects={'Attention': Attention})

    # Generate the prediction (this will be in the scaled space)
    y_pred_next = model.predict(X_next)

    # Prepare a padded array for inverse transformation.
    # We assume that the scaler was fit on the full feature set and that the target (price) is in the first column.
    y_pred_padded = np.zeros((1, scaler.min_.shape[0]))
    y_pred_padded[:, 0] = y_pred_next.flatten()

    # Inverse transform to get the prediction in original price scale
    next_day_price = scaler.inverse_transform(y_pred_padded)[:, 0][0]

    return next_day_price

def predict_xgboost(data, xgb_model_path, feature_subset):
    """
    Generate a prediction for the next trading day using the XGBoost model.

    This function:
      - Ensures 'Date' is present and formatted.
      - Computes technical indicators on the provided data.
      - Creates the 'Price_Change' and 'Target' columns.
      - Fills missing values using forward-fill, then drops rows missing critical features.
      - Uses the last available row to generate a prediction.
    
    Parameters:
    - data: pd.DataFrame, historical stock data.
    - xgb_model_path: str, path to the saved XGBoost .joblib model.
    - feature_subset: list of str, features used during model training.
    
    Returns:
    - signal: int, predicted signal (e.g. 1 for buy/up, 0 for sell/down).
    - prob_up: float, probability of an upward move.
    - pred_date: date, the date associated with the prediction.
    """
    import pandas as pd
    import numpy as np
    from joblib import load
    from data_processing import calculate_technical_indicators

    # Ensure 'Date' is present and properly formatted
    if 'Date' not in data.columns:
        data['Date'] = pd.to_datetime(data.index).date
    else:
        data['Date'] = pd.to_datetime(data['Date']).dt.date

    # Compute technical indicators
    data = calculate_technical_indicators(data)
    print("Columns after calculating indicators:", data.columns.tolist())

    # Create additional columns needed for prediction
    data['Price_Change'] = data['Close'].diff()
    data['Target'] = (data['Price_Change'] > 0).astype(int)

    # Determine critical features (features used for prediction plus target columns)
    critical_features = feature_subset + ['Price_Change', 'Target'] if feature_subset else ['Price_Change', 'Target']
    
    # Use forward fill to handle missing values, then drop any rows still missing critical features
    data = data.fillna(method='ffill')
    data = data.dropna(subset=critical_features)
    print("Data shape after forward fill and dropna:", data.shape)

    if data.empty:
        raise ValueError("DataFrame is empty after filling and dropping NaN values. Check your technical indicator calculations and input data.")

    # Select only the features used by the model
    X = data[feature_subset] if feature_subset else data.drop(columns=['Price_Change', 'Target'])

    # Load the saved XGBoost model
    model = load(xgb_model_path)
    print(f"Loaded XGBoost model from {xgb_model_path}")

    # Use the most recent row for prediction
    X_latest = X.tail(1)
    y_pred = model.predict(X_latest)
    y_pred_proba = model.predict_proba(X_latest)[:, 1]

    pred_date = data['Date'].iloc[-1]  
    # Get the corresponding date from the data for the last row
    next_trading_date = get_next_trading_day(pred_date)


    signal = y_pred[0]       # e.g., 1 for buy/up, 0 for sell/down
    prob_up = y_pred_proba[0]  # probability of an upward move

    return signal, prob_up, next_trading_date



def predict_nlp(ticker, window=15, days=30):
    """
    Generate a sentiment score prediction for the next trading day using your NLP approach.
    
    This function:
      - Sets a date range (default: last 30 days ending yesterday).
      - Calls generate_next_day_weighted_rolling_sentiments to compute weighted rolling sentiments.
      - Extracts the most recent prediction for the next day.
      - Converts the sentiment score into a trading signal (buy if score > 0, sell otherwise).
    
    Parameters:
    - ticker: str, the stock ticker symbol.
    - window: int, the window size used in sentiment calculation (default: 15).
    - days: int, how many days of historical data to consider (default: 30).
    
    Returns:
    - sentiment_score: float, the predicted sentiment score for the next trading day.
    - predicted_signal: str, the trading signal ("buy" if score > 0, "sell" otherwise).
    - pred_date: date, the date associated with the sentiment prediction.
    """
    import sys
    import os

    sys.path.append(os.path.join(os.path.dirname(__file__), 'nlp'))
    import datetime
    from run_nlp import generate_next_day_weighted_rolling_sentiments

    # Define the date range: from (end_date - days) to yesterday.
    today = datetime.date.today()
    end_date = today - datetime.timedelta(days=1)
    start_date = end_date - datetime.timedelta(days=days)

    # Generate the weighted rolling sentiments.
    # Note: generate_next_day_weighted_rolling_sentiments should return a DataFrame
    # with at least the columns "Date" and "next_day_sentiment".
    sentiment_df = generate_next_day_weighted_rolling_sentiments(ticker, str(start_date), str(end_date), window=window)
    
    if sentiment_df is None or sentiment_df.empty:
        raise ValueError("No sentiment data returned from NLP model.")
    
    # Extract the most recent prediction row.
    last_row = sentiment_df.iloc[-1]
    sentiment_score = last_row['next_day_sentiment']
    predicted_signal = "buy" if sentiment_score > 0 else "sell"
    pred_date = last_row['Date']  # This should be the date associated with the sentiment prediction.
    
    return sentiment_score, predicted_signal, pred_date


def predict_rl(stock_data, dqn_model_path, xgb_model_path, feature_subset):
    """
    Generate an RL trading signal for the next trading day using a trained DQN agent.
    
    This function:
      - Ensures the stock data has a properly formatted 'Date' column.
      - Uses the XGBoost model to compute predictions and merges them into the stock data.
      - Creates a trading environment (StockTradingEnv) using the merged data.
      - Loads the trained DQN model into a DQNAgent.
      - Simulates the environment through the historical data to reach a final state.
      - Queries the agent for its action on the final state, maps that to a signal, and computes the next trading day.
    
    Parameters:
      stock_data: pd.DataFrame of raw stock data.
      dqn_model_path: str, path to the saved DQN model.
      xgb_model_path: str, path to the saved XGBoost model.
      feature_subset: list of str, features used during XGBoost training.
    
    Returns:
      predicted_signal: str, e.g., "Buy", "Hold", or "Sell".
      next_date: date, the next trading day's date.
    """

    import sys
    import os

    sys.path.append(os.path.join(os.path.dirname(__file__), 'rl'))

    import pandas as pd
    import tensorflow as tf
    from datetime import timedelta
    from dqn_agent import DQNAgent
    from trading_env import StockTradingEnv

    # Ensure 'Date' exists and is properly formatted.
    if 'Date' not in stock_data.columns:
        stock_data['Date'] = pd.to_datetime(stock_data.index).date
    else:
        stock_data['Date'] = pd.to_datetime(stock_data['Date']).dt.date
    stock_data = stock_data.reset_index(drop=True)

    # Merge in XGBoost predictions.
    # Use your existing function to generate a DataFrame with columns like 'Date', 'XGB_Pred', 'XGB_Prob_Up'
    sys.path.append(os.path.join(os.path.dirname(__file__), 'boost'))
    from run_boost import xgboost_inference_df_from_df
    df_xgb = xgboost_inference_df_from_df(stock_data.copy(), xgb_model_path, feature_subset)
    merged_data = pd.merge(stock_data, df_xgb, on='Date', how='inner')
    
    # Check that the merged DataFrame has the required columns.
    if 'XGB_Pred' not in merged_data.columns:
        raise ValueError("Merged data does not contain 'XGB_Pred'. Ensure XGBoost predictions are being generated correctly.")
    
    # Create the trading environment using the merged data.
    env = StockTradingEnv(merged_data)
    agent = DQNAgent(state_size=env.state_size, action_size=len(env.action_space))
    
    # Load the trained DQN model.
    def custom_mse(y_true, y_pred):
        return tf.reduce_mean(tf.square(y_pred - y_true))
    custom_objects = {"mse": custom_mse}
    model = tf.keras.models.load_model(dqn_model_path, custom_objects=custom_objects)
    agent.model = model

    # Simulate the environment until done.
    state = env.reset()
    done = False
    while not done:
        action = agent.act(state)
        next_state, reward, done, info = env.step(action)
        state = next_state

    # Query the agent on the final state.
    predicted_action = agent.act(state)

    # Map the numeric action to a trading signal (adjust this mapping as needed).
    action_mapping = {0: "Sell", 1: "Hold", 2: "Buy"}
    predicted_signal = action_mapping.get(predicted_action, "Unknown")

    # Determine the next trading day based on the last date in the merged data.
    last_date = merged_data['Date'].max()
    next_date = last_date + timedelta(days=1)
    while next_date.weekday() >= 5:  # Skip Saturday (5) and Sunday (6)
        next_date += timedelta(days=1)

    return predicted_signal, next_date



# -------------------------------
# 3. Main Pipeline Function
# -------------------------------

def main():
    # Define parameters and paths
    ticker = "KO"
    lstm_model_path = "models/lstm/lstm_KO_model.h5"
    # Uncomment and add paths for your other models as needed:
    xgb_model_path = "models/boost/xgboost_KO.joblib"
    dqn_model_path = "models/rl/ko_dqn.h5"
    
    # Feature subset used by the XGBoost model (if applicable)
    feature_subset = [
        'WCLPRICE', 'BB_lower', 'APO', 'MEDPRICE', 'HT_DCPERIOD', 'TYPPRICE',
        'TRIMA', 'MACD_hist', 'T3', 'SMA', 'AVGPRICE', 'TRANGE', 'ADXR',
        'HT_TRENDMODE', 'STOCH_fastk', 'STOCH_slowk', 'STOCH_slowd', 'TEMA',
        'CMO', 'STOCH_fastd', 'HT_DCPHASE', 'AROON_DOWN', 'CCI', 'MFI', 'OBV',
        'MACD_signal', 'MINUS_DI', 'HT_LEADSINE', 'HT_PHASOR_inphase', 'WMA'
    ]
    
    # Step 1: Get the latest historical data
    data = get_latest_data(ticker)
    
    # Determine the last date in the dataset and compute the next trading day
    last_date = max(data['Date'])
    next_trading_date = get_next_trading_day(last_date)
    
    # Step 2: Generate predictions from each model
    lstm_price = predict_lstm(ticker, lstm_model_path, data)
    # Uncomment the lines below when you integrate your other models:
    xgb_signal, xgb_prob, xgb_date = predict_xgboost(data, xgb_model_path, feature_subset)
    nlp_sentiment, predicted_signal, nlp_date = predict_nlp(ticker)
    rl_signal, rl_date = predict_rl(data, dqn_model_path, xgb_model_path, feature_subset)
    
    # Step 3: Print/aggregate the predictions for the next trading day
    print("Predictions for the next trading day:")
    print(f" Date: {next_trading_date}")
    print(f" - LSTM predicted price: {lstm_price:.2f}")
    # Uncomment when other models are integrated:
    print(f" - XGBoost signal: {xgb_signal} (Probability up: {xgb_prob:.2f}) for date: {xgb_date}")
    print(f" - NLP sentiment score: {nlp_sentiment:.2f} signifying a {predicted_signal} signal for date: {nlp_date}")
    print(f" - RL signal: {rl_signal} for {rl_date}")

# -------------------------------
# 4. Scheduling Note
# -------------------------------
# To run this script automatically after the market closes, you could schedule it with:
# - A cron job (on Unix-like systems)
# - APScheduler within a Python application
# For example, with cron:
#
# 30 18 * * 1-5 /usr/bin/python3 /path/to/daily_prediction_pipeline.py
#
# This would run the script at 6:30 PM on weekdays (adjust the time as needed).

if __name__ == "__main__":
    main()
