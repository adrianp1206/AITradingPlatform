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
    plot=True,
    initial_balance=100_000   # <--- Assume environment starts with 100k
):
    """
    Run a trained DQN agent on 'merged' data and return both the detailed
    log (df_results) and summary metrics (final P/L, percent gain, annualized gain).
    """

    # ---------------------------------------------------------
    # 1) Fetch base OHLCV data
    # ---------------------------------------------------------
    df_base = fetch_stock_data_alpha(ticker, start_date=start_date, end_date=end_date, api_key=api_key)
    if 'Date' not in df_base.columns:
        df_base['Date'] = df_base.index
    df_base['Date'] = pd.to_datetime(df_base['Date']).dt.date
    df_base = df_base.reset_index(drop=True)

    # ---------------------------------------------------------
    # 2) Generate LSTM predictions
    # ---------------------------------------------------------
    df_lstm_raw, scaler = preprocess_data_alpha(df_base.copy())
    X_lstm, _ = create_lstm_input(df_lstm_raw.copy(), target_column='Close', lookback=20)
    model_lstm = load_model(lstm_model_path, custom_objects={'Attention': Attention})
    lstm_preds = model_lstm.predict(X_lstm)

    # Unscale LSTM predictions
    lstm_pred_padded = np.zeros((len(lstm_preds), scaler.min_.shape[0]))
    lstm_pred_padded[:, 0] = lstm_preds.flatten()
    lstm_unscaled = scaler.inverse_transform(lstm_pred_padded)[:, 0]

    # Align LSTM predictions with correct rows
    df_lstm = df_base.iloc[-len(lstm_unscaled):].copy()
    df_lstm["LSTM_Pred"] = lstm_unscaled

    # ---------------------------------------------------------
    # 3) Generate sentiment data
    # ---------------------------------------------------------
    sentiment_df = generate_next_day_weighted_rolling_sentiments(ticker, start_date, end_date)
    sentiment_df = sentiment_df.rename(columns={"next_day_sentiment": "Sentiment_Score"})
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

    model_xgb = load(xgb_model_path)
    if xgb_features is None:
        raise ValueError("Please provide xgb_features (list of features) for XGBoost.")
    X_xgb = df_xgb[xgb_features].dropna()

    # Filter df_xgb to rows that have no NaN in features
    df_xgb = df_xgb.loc[X_xgb.index]
    df_xgb["XGB_Pred"] = model_xgb.predict(X_xgb)
    # For classification: predict_proba -> probability that next move is "Up"
    df_xgb["XGB_Prob_Up"] = model_xgb.predict_proba(X_xgb)[:, 1]
    df_xgb = df_xgb.reset_index(drop=True)

    # ---------------------------------------------------------
    # 5) Merge everything into ONE DataFrame
    # ---------------------------------------------------------
    merged = pd.merge(
        df_lstm[["Date", "LSTM_Pred"]],
        df_base,
        on="Date", 
        how="inner"
    )
    merged = pd.merge(merged, sentiment_df[["Date", "Sentiment_Score"]], on="Date", how="inner")
    merged = pd.merge(merged, df_xgb[["Date", "XGB_Pred", "XGB_Prob_Up"]], on="Date", how="inner")

    # ---------------------------------------------------------
    # 6) Create the environment and the agent
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
        current_idx = min(env.current_step, len(env.df) - 1)
        date_val = env.df.iloc[current_idx]["Date"]
        close_val = env.df.iloc[current_idx - 1]["Close"] if current_idx > 0 else np.nan

        next_state, reward, done, info = env.step(action)
        cumulative_reward += reward

        logs.append({
            "Date": date_val,
            "Step": env.current_step,
            "Action": action,
            "Reward": reward,
            "Cumulative_Reward": cumulative_reward,
            "Realized_Profit": info["realized_profit"],  # from env
            "Balance": info["balance"],                  # from env
            "Close": close_val
        })
        state = next_state

    df_results = pd.DataFrame(logs)

    # ---------------------------------------------------------
    # 9) Compute Final Metrics
    # ---------------------------------------------------------
    # We'll take final realized profit from the environment logs
    final_profit = df_results["Realized_Profit"].iloc[-1] if not df_results.empty else 0.0
    final_balance = initial_balance + final_profit

    # Calculate time span in years
    start_dt = pd.to_datetime(start_date)
    end_dt = pd.to_datetime(end_date)
    days_diff = (end_dt - start_dt).days
    years_diff = days_diff / 365.25 if days_diff > 0 else 1.0

    percent_gain = (final_profit / initial_balance) * 100.0
    # Basic CAGR formula: (1 + raw_gain)^(1/years) - 1
    # raw_gain = final_profit / initial_balance
    raw_gain = final_profit / initial_balance
    annualized_gain = ((1 + raw_gain) ** (1 / years_diff) - 1) * 100.0 if raw_gain > -1 else -100.0

    # Print them
    print(f"\n--- {ticker} ---")
    print(f"Final Realized P/L: ${final_profit:,.2f}")
    print(f"Percent Gain: {percent_gain:.2f}%")
    print(f"Annualized Gain: {annualized_gain:.2f}%")
    print(f"Final Balance: ${final_balance:,.2f}")

    # Optionally plot the Cumulative Reward
    if plot and not df_results.empty:
        plt.figure(figsize=(10, 5))
        plt.plot(df_results["Date"], df_results["Cumulative_Reward"], label="Cumulative Reward")
        plt.xticks(rotation=45)
        plt.xlabel("Date")
        plt.ylabel("Cumulative Reward")
        plt.title(f"{ticker} RL Backtest (DQN) - Cumulative Reward")
        plt.legend()
        plt.tight_layout()
        plt.show()

    # Package summary metrics in a dict
    summary_metrics = {
        "ticker": ticker,
        "final_profit": final_profit,
        "final_balance": final_balance,
        "percent_gain": percent_gain,
        "annualized_gain": annualized_gain
    }

    # Return both detailed logs and summary
    return df_results, summary_metrics

def run_trained_agent_on_data_long(
    ticker,
    start_date,
    end_date,
    api_key,
    dqn_model_path,
    xgb_model_path,
    lstm_model_path,
    xgb_features=None,
    plot=True,
    initial_balance=100_000   # <--- Assume environment starts with 100k
):
    """
    Run a trained DQN agent on 'merged' data and return both the detailed
    log (df_results) and summary metrics (final P/L, percent gain, annualized gain).
    """

    # ---------------------------------------------------------
    # 1) Fetch base OHLCV data
    # ---------------------------------------------------------
    df_base = fetch_stock_data_alpha(ticker, start_date=start_date, end_date=end_date, api_key=api_key)
    if 'Date' not in df_base.columns:
        df_base['Date'] = df_base.index
    df_base['Date'] = pd.to_datetime(df_base['Date']).dt.date
    df_base = df_base.reset_index(drop=True)

    # ---------------------------------------------------------
    # 2) Generate LSTM predictions
    # ---------------------------------------------------------
    df_lstm_raw, scaler = preprocess_data_alpha(df_base.copy())
    X_lstm, _ = create_lstm_input(df_lstm_raw.copy(), target_column='Close', lookback=20)
    model_lstm = load_model(lstm_model_path, custom_objects={'Attention': Attention})
    lstm_preds = model_lstm.predict(X_lstm)

    # Unscale LSTM predictions
    lstm_pred_padded = np.zeros((len(lstm_preds), scaler.min_.shape[0]))
    lstm_pred_padded[:, 0] = lstm_preds.flatten()
    lstm_unscaled = scaler.inverse_transform(lstm_pred_padded)[:, 0]

    # Align LSTM predictions with correct rows
    df_lstm = df_base.iloc[-len(lstm_unscaled):].copy()
    df_lstm["LSTM_Pred"] = lstm_unscaled

    # ---------------------------------------------------------
    # 3) Generate sentiment data
    # ---------------------------------------------------------
    sentiment_df = generate_next_day_weighted_rolling_sentiments(ticker, start_date, end_date)
    sentiment_df = sentiment_df.rename(columns={"next_day_sentiment": "Sentiment_Score"})
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

    model_xgb = load(xgb_model_path)
    if xgb_features is None:
        raise ValueError("Please provide xgb_features (list of features) for XGBoost.")
    X_xgb = df_xgb[xgb_features].dropna()

    # Filter df_xgb to rows that have no NaN in features
    df_xgb = df_xgb.loc[X_xgb.index]
    df_xgb["XGB_Pred"] = model_xgb.predict(X_xgb)
    # For classification: predict_proba -> probability that next move is "Up"
    df_xgb["XGB_Prob_Up"] = model_xgb.predict_proba(X_xgb)[:, 1]
    df_xgb = df_xgb.reset_index(drop=True)

    # ---------------------------------------------------------
    # 5) Merge everything into ONE DataFrame
    # ---------------------------------------------------------
    merged = pd.merge(
        df_lstm[["Date", "LSTM_Pred"]],
        df_base,
        on="Date", 
        how="inner"
    )
    merged = pd.merge(merged, sentiment_df[["Date", "Sentiment_Score"]], on="Date", how="inner")
    merged = pd.merge(merged, df_xgb[["Date", "XGB_Pred", "XGB_Prob_Up"]], on="Date", how="inner")

    # ---------------------------------------------------------
    # 6) Create the environment and the agent
    # ---------------------------------------------------------
    env = StockTradingEnvLongOnly(merged)
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
        current_idx = min(env.current_step, len(env.df) - 1)
        date_val = env.df.iloc[current_idx]["Date"]
        close_val = env.df.iloc[current_idx - 1]["Close"] if current_idx > 0 else np.nan

        next_state, reward, done, info = env.step(action)
        cumulative_reward += reward

        logs.append({
            "Date": date_val,
            "Step": env.current_step,
            "Action": action,
            "Reward": reward,
            "Cumulative_Reward": cumulative_reward,
            "Realized_Profit": info["realized_profit"],  # from env
            "Balance": info["balance"],                  # from env
            "Close": close_val
        })
        state = next_state

    df_results = pd.DataFrame(logs)

    # ---------------------------------------------------------
    # 9) Compute Final Metrics
    # ---------------------------------------------------------
    # We'll take final realized profit from the environment logs
    final_profit = df_results["Realized_Profit"].iloc[-1] if not df_results.empty else 0.0
    final_balance = initial_balance + final_profit

    # Calculate time span in years
    start_dt = pd.to_datetime(start_date)
    end_dt = pd.to_datetime(end_date)
    days_diff = (end_dt - start_dt).days
    years_diff = days_diff / 365.25 if days_diff > 0 else 1.0

    percent_gain = (final_profit / initial_balance) * 100.0
    # Basic CAGR formula: (1 + raw_gain)^(1/years) - 1
    # raw_gain = final_profit / initial_balance
    raw_gain = final_profit / initial_balance
    annualized_gain = ((1 + raw_gain) ** (1 / years_diff) - 1) * 100.0 if raw_gain > -1 else -100.0

    # Print them
    print(f"\n--- {ticker} ---")
    print(f"Final Realized P/L: ${final_profit:,.2f}")
    print(f"Percent Gain: {percent_gain:.2f}%")
    print(f"Annualized Gain: {annualized_gain:.2f}%")
    print(f"Final Balance: ${final_balance:,.2f}")

    # Optionally plot the Cumulative Reward
    if plot and not df_results.empty:
        plt.figure(figsize=(10, 5))
        plt.plot(df_results["Date"], df_results["Cumulative_Reward"], label="Cumulative Reward")
        plt.xticks(rotation=45)
        plt.xlabel("Date")
        plt.ylabel("Cumulative Reward")
        plt.title(f"{ticker} RL Backtest (DQN) - Cumulative Reward")
        plt.legend()
        plt.tight_layout()
        plt.show()

    # Package summary metrics in a dict
    summary_metrics = {
        "ticker": ticker,
        "final_profit": final_profit,
        "final_balance": final_balance,
        "percent_gain": percent_gain,
        "annualized_gain": annualized_gain
    }

    # Return both detailed logs and summary
    return df_results, summary_metrics
