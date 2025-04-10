import sys
import os
import pandas as pd
import numpy as np
from joblib import load
from tensorflow.keras.models import load_model

# sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
# from data_processing import (
#     fetch_stock_data_alpha,
#     preprocess_data_alpha,
#     create_lstm_input,
#     calculate_technical_indicators
# )
# sys.path.append(os.path.join(os.path.dirname(__file__), '../lstm'))
# from lstm_model import Attention
# sys.path.append(os.path.join(os.path.dirname(__file__), '../nlp'))
# from run_nlp import generate_next_day_weighted_rolling_sentiments
from trading_env_shares import StockTradingEnv, StockTradingEnvLongOnly
from dqn_agent import DQNAgent
from train_rl import train_dqn_agent

def build_rl_input_csv(ticker, start_date, end_date, api_key, lstm_model_path, xgb_model_path, xgb_features, output_path):
    df_base = fetch_stock_data_alpha(ticker, start_date=start_date, end_date=end_date, api_key=api_key)
    df_base = df_base.reset_index()
    df_base.rename(columns={df_base.columns[0]: "Date"}, inplace=True)

    df_lstm_raw, scaler = preprocess_data_alpha(df_base.copy())
    X_lstm, _ = create_lstm_input(df_lstm_raw.copy(), target_column='Close', lookback=20)
    model_lstm = load_model(lstm_model_path, custom_objects={'Attention': Attention})
    lstm_preds = model_lstm.predict(X_lstm)

    lstm_pred_padded = np.zeros((len(lstm_preds), scaler.min_.shape[0]))
    lstm_pred_padded[:, 0] = lstm_preds.flatten()
    lstm_unscaled = scaler.inverse_transform(lstm_pred_padded)[:, 0]

    df_lstm = df_base.iloc[-len(lstm_unscaled):].copy()
    df_lstm["LSTM_Pred"] = lstm_unscaled

    sentiment_df = generate_next_day_weighted_rolling_sentiments(ticker, start_date, end_date)
    sentiment_df = sentiment_df.rename(columns={"next_day_sentiment": "Sentiment_Score"})

    df_xgb = fetch_stock_data_alpha(ticker, start_date=start_date, end_date=end_date, api_key=api_key)
    df_xgb = calculate_technical_indicators(df_xgb)
    df_xgb = df_xgb.reset_index()
    df_xgb.rename(columns={df_xgb.columns[0]: "Date"}, inplace=True)

    model_xgb = load(xgb_model_path)
    X_xgb = df_xgb[xgb_features].dropna()
    df_xgb = df_xgb.loc[X_xgb.index]
    df_xgb["XGB_Pred"] = model_xgb.predict(X_xgb)
    df_xgb["XGB_Prob_Up"] = model_xgb.predict_proba(X_xgb)[:, 1]
    df_xgb = df_xgb.reset_index(drop=True)

    merged = pd.merge(df_lstm[["Date", "LSTM_Pred"]], sentiment_df[["Date", "Sentiment_Score"]], on="Date", how="inner")
    merged = pd.merge(merged, df_xgb[["Date", "XGB_Pred", "XGB_Prob_Up"]], on="Date", how="inner")
    merged = pd.merge(merged, df_base, on="Date", how="inner")

    merged.to_csv(output_path, index=False)
    print(f"✅ RL input saved to {output_path}")

def train_rl_from_csv(input_csv, episodes=10, save_path_1=None, save_path_2=None):
    import os
    from google.colab import files
    
    df = pd.read_csv(input_csv)

    env = StockTradingEnv(df, initial_balance=10000)
    agent = DQNAgent(state_size=env.state_size, action_size=len(env.action_space))
    train_dqn_agent(agent, env, episodes=episodes, save_path=save_path_1)

    # env = StockTradingEnvLongOnly(df, initial_balance=10000)
    # agent = DQNAgent(state_size=env.state_size, action_size=len(env.action_space))
    # train_dqn_agent(agent, env, episodes=episodes, save_path=save_path_2)

    if save_path_1 and os.path.exists(save_path_1):
        print(f"📥 Downloading {save_path_1}")
        files.download(save_path_1)

    # if save_path_2 and os.path.exists(save_path_2):
    #     print(f"📥 Downloading {save_path_2}")
    #     files.download(save_path_2)

if __name__ == "__main__":
    # build_rl_input_csv(
    #     ticker="NKE",
    #     start_date="2021-05-01",
    #     end_date="2024-04-01",
    #     api_key="NL9PDOM5JWRPAT9O",
    #     lstm_model_path="../models/lstm/lstm_NKE_model.h5",
    #     xgb_model_path="../models/boost/xgboost_NKE.joblib",
    #     xgb_features=['NATR', 'HT_DCPHASE', 'APO', 'CMO', 'HT_TRENDMODE', 'STOCH_fastd', 'MIDPOINT', 'MACD', 'HT_PHASOR_quadrature', 'MFI', 'AROON_UP', 'RSI', 'AROON_DOWN', 'AROONOSC', 'STOCH_slowd', 'TRANGE', 'SMA', 'TEMA', 'HT_SINE', 'PLUS_DI', 'BB_lower', 'STOCH_fastk', 'MIDPRICE', 'BB_upper', 'ADX', 'STOCH_slowk', 'MEDPRICE', 'MINUS_DI'],
    #     output_path="NKE_RL_input.csv"
    # )

    train_rl_from_csv(
        input_csv="KO_RL_input.csv",
        episodes=5,
        save_path_1="ko_dqn_model_both.h5",
        save_path_2="ko_dqn_model_long.h5"
    )
