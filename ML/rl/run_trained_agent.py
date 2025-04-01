# Paths setup
import sys
import os

ml_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))

# Add ML root to sys.path so we can import data_processing.py
if ml_root not in sys.path:
    sys.path.append(ml_root)

# Add boost and lstm directories to sys.path if needed
boost_path = os.path.join(ml_root, 'boost')
lstm_path = os.path.join(ml_root, 'lstm')

for path in [boost_path, lstm_path]:
    if path not in sys.path:
        sys.path.append(path)

# Now imports will work
from data_processing import fetch_stock_data_alpha
from run_boost import xgboost_inference_df_from_df
# from run_lstm import generate_lstm_predictions_from_df

import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt

from dqn_agent import DQNAgent
from trading_env import StockTradingEnv

def run_trained_agent_on_data(
    ticker,
    start_date,
    end_date,
    api_key,
    dqn_model_path,
    xgb_model_path,
    lstm_model_path,
    feature_subset=None,
    plot=True
):
    """
    Run a trained DQN agent on fresh market data (no CSV needed).
    """
    # Step 1: Fetch stock data
    stock_df = fetch_stock_data_alpha(ticker, api_key=api_key, start_date=start_date, end_date=end_date)

    # Ensure 'Date' column exists
    if 'Date' not in stock_df.columns:
        stock_df['Date'] = stock_df.index

    stock_df['Date'] = pd.to_datetime(stock_df['Date']).dt.date
    stock_df = stock_df.reset_index(drop=True)

    # Step 2: Run XGBoost predictions using in-memory DataFrame version
    df_xgb = xgboost_inference_df_from_df(stock_df.copy(), xgb_model_path, feature_subset)

    # Step 3: Merge stock data and model predictions
    df_merged = pd.merge(stock_df, df_xgb, on='Date', how='inner')

    print(stock_df['Date'].dtype, stock_df['Date'].head())
    print(df_xgb['Date'].dtype, df_xgb['Date'].head())
    print("Merged shape:", df_merged.shape)

    # Step 4: Set up environment
    env = StockTradingEnv(df_merged)
    agent = DQNAgent(state_size=env.state_size, action_size=len(env.action_space))

    # Step 5: Load the trained model
    def custom_mse(y_true, y_pred):
        return tf.reduce_mean(tf.square(y_pred - y_true))

    custom_objects = {"mse": custom_mse}
    model = tf.keras.models.load_model(dqn_model_path, custom_objects=custom_objects)
    agent.model = model

    # Step 6: Backtest loop
    state = env.reset()
    done = False
    logs = []
    cumulative_reward = 0

    while not done:
        action = agent.act(state)

        # Safely log current info before env.step() bumps the index
        current_idx = min(env.current_step, len(env.df) - 1)
        date_val = env.df.iloc[current_idx]["Date"]
        close_val = env.df.iloc[current_idx - 1]["Close"] if current_idx > 0 else None

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

    # Step 7: Wrap up
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

        print("REALIZED PROFIT:")
        print(df_results["Realized_Profit"].iloc[-1])

    return df_results
