import sys
import os
import pandas as pd
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model
from joblib import load  # For XGBoost model

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from data_processing import fetch_stock_data_alpha, calculate_technical_indicators, preprocess_data_alpha, create_lstm_input
from dqn_agent import DQNAgent

sys.path.append(os.path.join(os.path.dirname(__file__), '../lstm'))
from lstm_model import Attention
sys.path.append(os.path.join(os.path.dirname(__file__), '../nlp'))
from run_nlp import generate_next_day_weighted_rolling_sentiments

from trading_env_shares import StockTradingEnv, StockTradingEnvLongOnly

def run_trained_agent_on_data(
    ticker,
    start_date,
    end_date,
    api_key,
    dqn_model_path,
    xgb_model_path,
    lstm_model_path,
    xgb_features=None,
    plot=True
):
    """
    Run a trained DQN agent on 'merged' data that includes:
      - Raw OHLCV data
      - LSTM predictions
      - XGBoost predictions
      - Sentiment scores
    for a proper backtest.
    """

    # ---------------------------------------------------------
    # 1) Fetch base OHLCV data
    # ---------------------------------------------------------
    df_base = fetch_stock_data_alpha(ticker, start_date=start_date, end_date=end_date, api_key=api_key)
    # Ensure 'Date' is a proper column
    if 'Date' not in df_base.columns:
        df_base['Date'] = df_base.index
    df_base['Date'] = pd.to_datetime(df_base['Date']).dt.date
    df_base = df_base.reset_index(drop=True)

    # ---------------------------------------------------------
    # 2) Generate LSTM predictions (replicating build_rl_input_csv steps)
    # ---------------------------------------------------------
    # a) Preprocess data for LSTM
    df_lstm_raw, scaler = preprocess_data_alpha(df_base.copy())
    # b) Create LSTM input
    X_lstm, _ = create_lstm_input(df_lstm_raw.copy(), target_column='Close', lookback=20)
    model_lstm = load_model(lstm_model_path, custom_objects={'Attention': Attention})
    lstm_preds = model_lstm.predict(X_lstm)

    # d) "Unscale" the LSTM predictions
    #    The shape of 'scaler.min_' or 'scaler.data_min_' depends on your scaler
    lstm_pred_padded = np.zeros((len(lstm_preds), scaler.min_.shape[0]))
    lstm_pred_padded[:, 0] = lstm_preds.flatten()
    lstm_unscaled = scaler.inverse_transform(lstm_pred_padded)[:, 0]

    # e) Align LSTM predictions with the correct dates
    df_lstm = df_base.iloc[-len(lstm_unscaled):].copy()
    df_lstm["LSTM_Pred"] = lstm_unscaled

    # ---------------------------------------------------------
    # 3) Generate sentiment data
    # ---------------------------------------------------------
    sentiment_df = generate_next_day_weighted_rolling_sentiments(ticker, start_date, end_date)
    sentiment_df = sentiment_df.rename(columns={"next_day_sentiment": "Sentiment_Score"})
    # Make sure 'Date' in sentiment_df is of the same date format
    sentiment_df["Date"] = pd.to_datetime(sentiment_df["Date"]).dt.date

    # ---------------------------------------------------------
    # 4) Generate XGBoost predictions
    # ---------------------------------------------------------
    df_xgb = fetch_stock_data_alpha(ticker, start_date=start_date, end_date=end_date, api_key=api_key)
    df_xgb = calculate_technical_indicators(df_xgb)
    if 'Date' not in df_xgb.columns:
        df_xgb['Date'] = df_xgb.index
    df_xgb['Date'] = pd.to_datetime(df_xgb['Date']).dt.date
    df_xgb = df_xgb.reset_index(drop=True)

    model_xgb = load(xgb_model_path)  # Load XGB model
    # Use the specified XGB features (like feature_subset in your old code)
    if xgb_features is None:
        raise ValueError("Please provide xgb_features (list of features) for XGBoost.")
    X_xgb = df_xgb[xgb_features].dropna()

    # Filter df_xgb to rows that have no NaN in the features
    df_xgb = df_xgb.loc[X_xgb.index]
    df_xgb["XGB_Pred"] = model_xgb.predict(X_xgb)
    # If your XGB model is binary classification, you can do predict_proba
    df_xgb["XGB_Prob_Up"] = model_xgb.predict_proba(X_xgb)[:, 1]
    df_xgb = df_xgb.reset_index(drop=True)

    # ---------------------------------------------------------
    # 5) Merge everything into ONE DataFrame
    # ---------------------------------------------------------
    # Merge LSTM predictions
    merged = pd.merge(
        df_lstm[["Date", "LSTM_Pred"]],
        df_base,
        on="Date", 
        how="inner"
    )
    # Merge sentiment
    merged = pd.merge(merged, sentiment_df[["Date", "Sentiment_Score"]], on="Date", how="inner")
    # Merge XGB
    merged = pd.merge(merged, df_xgb[["Date", "XGB_Pred", "XGB_Prob_Up"]], on="Date", how="inner")

    # merged now contains: [Date, OHLCV columns, LSTM_Pred, Sentiment_Score, XGB_Pred, XGB_Prob_Up, ...]

    # ---------------------------------------------------------
    # 6) Create StockTradingEnv with merged data
    # ---------------------------------------------------------
    env = StockTradingEnv(merged)
    agent = DQNAgent(state_size=env.state_size, action_size=len(env.action_space))

    # ---------------------------------------------------------
    # 7) Load the trained DQN model
    # ---------------------------------------------------------
    def custom_mse(y_true, y_pred):
        return tf.reduce_mean(tf.square(y_pred - y_true))

    custom_objects = {"mse": custom_mse}
    model = tf.keras.models.load_model(dqn_model_path, custom_objects=custom_objects)
    agent.model = model

    # ---------------------------------------------------------
    # 8) Run the Backtest Loop
    # ---------------------------------------------------------
    state = env.reset()
    done = False
    logs = []
    cumulative_reward = 0

    while not done:
        action = agent.act(state)
        
        # Log info before stepping
        current_idx = min(env.current_step, len(env.df) - 1)
        date_val = env.df.iloc[current_idx]["Date"]
        close_val = (
            env.df.iloc[current_idx - 1]["Close"] if current_idx > 0 else np.nan
        )

        next_state, reward, done, info = env.step(action)
        cumulative_reward += reward

        logs.append({
            "Date": date_val,
            "Step": env.current_step,
            "Action": action,
            "Reward": reward,
            "Cumulative_Reward": cumulative_reward,
            "Realized_Profit": info["realized_profit"],
            "Balance": info["balance"],
            "Close": close_val
        })

        state = next_state

    # ---------------------------------------------------------
    # 9) Visualization / Return results
    # ---------------------------------------------------------
    df_results = pd.DataFrame(logs)

    if plot:
        plt.figure(figsize=(10, 6))
        plt.plot(df_results["Date"], df_results["Cumulative_Reward"], label="Cumulative Reward")
        plt.xticks(rotation=45)
        plt.xlabel("Date")
        plt.ylabel("Cumulative Reward")
        plt.title(f"{ticker} RL Backtest (DQN) - Cumulative Reward")
        plt.legend()
        plt.tight_layout()
        plt.show()

        print("FINAL REALIZED PROFIT:", df_results["Realized_Profit"].iloc[-1])

    return df_results
