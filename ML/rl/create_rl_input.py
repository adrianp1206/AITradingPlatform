import pandas as pd

from run_boost import xgboost_inference_df_from_csv      # for XGBoost predictions
from run_lstm import generate_lstm_predictions_from_csv    # for LSTM predictions

def generate_rl_input_data(
    raw_csv_path,
    ticker,
    lstm_model_path,
    xgb_model_path,
    start_date='2008-01-01',
    end_date='2022-12-31',
    feature_subset=None,
):
    # Load the raw CSV data
    stock_df = pd.read_csv(raw_csv_path)
    
    # Optionally reorder columns if your model expects a specific order.
    # Expected order used by your original function:
    expected_order = ["Date", "Close", "High", "Low", "Open", "Volume"]
    if set(expected_order).issubset(stock_df.columns):
        stock_df = stock_df[expected_order]
    else:
        print("Warning: CSV columns do not match expected order. Proceeding with original column order.")
    
    # Convert the Date column to datetime.date objects
    stock_df["Date"] = pd.to_datetime(stock_df["Date"]).dt.date
    stock_df = stock_df.reset_index(drop=True)
    
    # Convert start_date and end_date to date objects for proper comparison
    start_date = pd.to_datetime(start_date).date()
    end_date = pd.to_datetime(end_date).date()
    
    # Filter the stock data by the desired date range
    stock_df = stock_df[(stock_df["Date"] >= start_date) & (stock_df["Date"] <= end_date)]
    
    # Get XGBoost predictions DataFrame
    df_xgb = xgboost_inference_df_from_csv(raw_csv_path, xgb_model_path, start_date, end_date, feature_subset)
    # Convert its Date column to datetime.date if needed
    df_xgb["Date"] = pd.to_datetime(df_xgb["Date"]).dt.date
    
    df_rl = pd.merge(stock_df, df_xgb, on='Date', how='inner')
    
    # Reset index if desired
    df_rl = df_rl.reset_index(drop=True)
    
    return df_rl
