import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
from scipy.stats import pearsonr
from data_processing import fetch_stock_data_alpha  # Assumes returns a DataFrame with price columns including "Close"

def analyze_predictions(pred_csv, api_key, actual_start_date, actual_end_date):
    # Load predictions CSV – expected columns:
    # ticker, lstm_predicted_price, next_trading_date, nlp_sentiment_score,
    # rl_recommendation, updated_at, xgboost_prob, xgboost_signal
    preds = pd.read_csv(pred_csv)
    preds['next_trading_date'] = pd.to_datetime(preds['next_trading_date'])
    
    comparisons = []
    
    # Loop over each unique ticker in the predictions CSV.
    tickers = preds['ticker'].unique()
    for ticker in tickers:
        print(f"Processing ticker: {ticker}")
        pred_ticker = preds[preds['ticker'] == ticker].copy()
        
        # Fetch actual data for this ticker.
        actual_df = fetch_stock_data_alpha(ticker, api_key=api_key, start_date=actual_start_date, end_date=actual_end_date)
        
        # If there's no "Date" column, try to use the index or generate a date range.
        if 'Date' not in actual_df.columns:
            if isinstance(actual_df.index, pd.DatetimeIndex):
                actual_df = actual_df.reset_index().rename(columns={'index': 'Date'})
            else:
                business_days = pd.date_range(start=actual_start_date, end=actual_end_date, freq='B')
                if len(business_days) == len(actual_df):
                    actual_df['Date'] = business_days
                else:
                    raise ValueError("Row count in actual_df does not match number of business days in range.")
        
        actual_df['Date'] = pd.to_datetime(actual_df['Date'])
        actual_df = actual_df.sort_values(by='Date')
        
        # Process each prediction row for this ticker.
        for idx, row in pred_ticker.iterrows():
            pred_date = row['next_trading_date']
            prev_df = actual_df[actual_df['Date'] < pred_date]
            if prev_df.empty:
                continue
            prev_day = prev_df.iloc[-1]
            prev_close = prev_day['Close']
            
            # Get actual close on the prediction date.
            actual_row = actual_df[actual_df['Date'] == pred_date]
            if actual_row.empty:
                continue
            actual_close = actual_row.iloc[0]['Close']
            
            # Calculate the percentage return (decimal) for this trade.
            pct_return = (actual_close - prev_close) / prev_close
            
            # Derive signals:
            lstm_signal = "Up" if row['lstm_predicted_price'] > prev_close else "Down"
            xgboost_signal = row['xgboost_signal']
            if row['rl_recommendation'].strip().lower() == "buy":
                rl_signal = "Up"
            elif row['rl_recommendation'].strip().lower() == "sell":
                rl_signal = "Down"
            else:
                rl_signal = "Hold"
            
            def calc_profit(signal):
                if signal == "Up":
                    return actual_close - prev_close
                elif signal == "Down":
                    return prev_close - actual_close
                else:
                    return 0.0
            
            lstm_profit = calc_profit(lstm_signal)
            xgboost_profit = calc_profit(xgboost_signal)
            rl_profit = calc_profit(rl_signal)
            
            comparisons.append({
                "ticker": ticker,
                "next_trading_date": pred_date,
                "prev_close": prev_close,
                "actual_close": actual_close,
                "actual_return": pct_return,
                "lstm_predicted_price": row['lstm_predicted_price'],
                "lstm_signal": lstm_signal,
                "lstm_profit": lstm_profit,
                "xgboost_prob": row['xgboost_prob'],
                "xgboost_signal": xgboost_signal,
                "xgboost_profit": xgboost_profit,
                "rl_recommendation": row['rl_recommendation'],
                "rl_signal": rl_signal,
                "rl_profit": rl_profit,
                "nlp_sentiment_score": row['nlp_sentiment_score'],
                "updated_at": row['updated_at']
            })
    
    comp_df = pd.DataFrame(comparisons)
    # Add per-trade percentage return columns.
    comp_df['lstm_pct'] = (comp_df['lstm_profit'] / comp_df['prev_close']) * 100
    comp_df['xgboost_pct'] = (comp_df['xgboost_profit'] / comp_df['prev_close']) * 100
    comp_df['rl_pct'] = (comp_df['rl_profit'] / comp_df['prev_close']) * 100
    
    comp_df.to_csv("comparison_results.csv", index=False)
    print(f"Saved {len(comp_df)} comparison rows to comparison_results.csv")
    
    # ---------------------------
    # Compute Overall Model Performance Metrics (in dollars)
    # ---------------------------
    lstm_mae = np.mean(np.abs(comp_df['lstm_predicted_price'] - comp_df['actual_close']))
    lstm_rmse = np.sqrt(np.mean((comp_df['lstm_predicted_price'] - comp_df['actual_close']) ** 2))
    lstm_directional_accuracy = np.mean(np.sign(comp_df['lstm_profit']) == np.sign(comp_df['actual_return'])) * 100
    xgboost_directional_accuracy = np.mean(np.sign(comp_df['xgboost_profit']) == np.sign(comp_df['actual_return'])) * 100
    rl_trade_mask = comp_df['rl_signal'] != "Hold"
    if rl_trade_mask.sum() > 0:
        rl_directional_accuracy = np.mean(np.sign(comp_df.loc[rl_trade_mask, 'rl_profit']) == np.sign(comp_df.loc[rl_trade_mask, 'actual_return'])) * 100
    else:
        rl_directional_accuracy = None
    if len(comp_df) > 1:
        nlp_corr, _ = pearsonr(comp_df['nlp_sentiment_score'], comp_df['actual_return'])
    else:
        nlp_corr = None
    
    # Overall portfolio simulation per model (dollar profits)
    portfolio_summary = {}
    for model in ['lstm', 'xgboost', 'rl']:
        profit_col = f"{model}_profit"
        if model == "rl":
            mask = comp_df['rl_signal'] != "Hold"
            profits = comp_df.loc[mask, profit_col]
        else:
            profits = comp_df[profit_col]
        total_profit = profits.sum()
        sharpe = profits.mean() / profits.std() if profits.std() > 0 else 0
        portfolio_summary[model] = {
            "Total Profit ($)": total_profit,
            "Average Profit ($)": profits.mean(),
            "Sharpe Ratio": sharpe,
            "Number of Trades": len(profits)
        }
    
    # Overall cumulative percentage returns (dollar-based).
    overall_pct_lstm = (comp_df['lstm_profit'].sum() / comp_df['prev_close'].sum()) * 100
    overall_pct_xgboost = (comp_df['xgboost_profit'].sum() / comp_df['prev_close'].sum()) * 100
    cumulative_rl = comp_df.loc[comp_df['rl_signal'] != "Hold"]
    overall_pct_rl = (cumulative_rl['rl_profit'].sum() / cumulative_rl['prev_close'].sum()) * 100 if len(cumulative_rl) > 0 else 0
    
    # Calculate overall time period (in years) from the earliest to the latest prediction date.
    min_date = comp_df['next_trading_date'].min()
    max_date = comp_df['next_trading_date'].max()
    T_years = (max_date - min_date).days / 365.25 if (max_date - min_date).days > 0 else 1
    
    def annualize_return(cum_pct):
        # Convert cumulative % return from percent to decimal, then compute CAGR.
        r = cum_pct / 100.0
        ann = ( (1 + r) ** (1 / T_years) - 1 ) * 100
        return ann
    
    overall_annualized = {
        "LSTM Annualized Return (%)": annualize_return(overall_pct_lstm),
        "XGBOOST Annualized Return (%)": annualize_return(overall_pct_xgboost),
        "RL Annualized Return (%)": annualize_return(overall_pct_rl)
    }
    
    print("\n--- Overall Model Performance Metrics (Dollar) ---")
    print(f"LSTM MAE: {lstm_mae:.4f}, RMSE: {lstm_rmse:.4f}, Directional Accuracy: {lstm_directional_accuracy:.2f}%")
    print(f"XGBOOST Directional Accuracy: {xgboost_directional_accuracy:.2f}%")
    if rl_directional_accuracy is not None:
        print(f"RL Directional Accuracy (excluding Hold): {rl_directional_accuracy:.2f}%")
    if nlp_corr is not None:
        print(f"NLP Sentiment / Return Correlation: {nlp_corr:.4f}")
    
    print("\n--- Overall Portfolio Simulation per Model (Dollar Profits) ---")
    for model, metrics in portfolio_summary.items():
        print(f"{model.upper()}: {metrics}")
    
    print("\n--- Overall Portfolio Average Trade Percentage Returns ---")
    print(f"LSTM Average % Return: {comp_df['lstm_pct'].mean():.2f}%")
    print(f"XGBOOST Average % Return: {comp_df['xgboost_pct'].mean():.2f}%")
    avg_rl_pct = comp_df.loc[comp_df['rl_signal'] != "Hold", 'rl_pct'].mean()
    print(f"RL Average % Return (excluding Hold): {avg_rl_pct:.2f}%")
    
    print("\n--- Overall Cumulative Percentage Returns ---")
    print(f"LSTM Cumulative % Return: {overall_pct_lstm:.2f}%")
    print(f"XGBOOST Cumulative % Return: {overall_pct_xgboost:.2f}%")
    print(f"RL Cumulative % Return: {overall_pct_rl:.2f}%")
    
    print("\n--- Overall Annualized Returns ---")
    for model, ann_return in overall_annualized.items():
        print(f"{model}: {ann_return:.2f}%")
    
    print("\n--- Overall Combined Profit across Models (Dollar) ---")
    combined_profit_lstm = comp_df['lstm_profit'].sum()
    combined_profit_xgboost = comp_df['xgboost_profit'].sum()
    combined_profit_rl = comp_df.loc[comp_df['rl_signal'] != "Hold", 'rl_profit'].sum()
    overall_portfolio = {
        "LSTM ($)": combined_profit_lstm,
        "XGBOOST ($)": combined_profit_xgboost,
        "RL ($)": combined_profit_rl
    }
    print(overall_portfolio)
    
    # ---------------------------
    # Per-Stock Summary for Each Model (including % returns & Annualized Returns)
    # ---------------------------
    per_stock_summary = {}
    for ticker in comp_df['ticker'].unique():
        ticker_df = comp_df[comp_df['ticker'] == ticker]
        ticker_metrics = {}
        # Compute time period (in years) for this ticker
        ticker_min = ticker_df['next_trading_date'].min()
        ticker_max = ticker_df['next_trading_date'].max()
        T_stock = (ticker_max - ticker_min).days / 365.25 if (ticker_max - ticker_min).days > 0 else 1
        
        def annualize_return_stock(cum_pct):
            r = cum_pct / 100.0
            return ((1 + r) ** (1 / T_stock) - 1) * 100
        
        for model in ['lstm', 'xgboost', 'rl']:
            profit_col = f"{model}_profit"
            pct_col = f"{model}_pct"
            if model == "rl":
                model_df = ticker_df[ticker_df['rl_signal'] != "Hold"]
            else:
                model_df = ticker_df
            total_profit = model_df[profit_col].sum()
            avg_profit = model_df[profit_col].mean() if not model_df.empty else np.nan
            std_profit = model_df[profit_col].std() if not model_df.empty else np.nan
            sharpe = avg_profit / std_profit if std_profit and std_profit > 0 else 0
            avg_pct = model_df[pct_col].mean() if not model_df.empty else np.nan
            cumulative_pct = (model_df[profit_col].sum() / model_df['prev_close'].sum()) * 100 if model_df['prev_close'].sum() != 0 else np.nan
            annualized_return = annualize_return_stock(cumulative_pct)
            directional_accuracy = (np.mean(np.sign(model_df[profit_col]) == np.sign(model_df['actual_return'])) * 100
                                    if not model_df.empty else np.nan)
            ticker_metrics[model] = {
                "Total Profit ($)": total_profit,
                "Average Profit ($)": avg_profit,
                "Sharpe Ratio": sharpe,
                "Average % Return": avg_pct,
                "Cumulative % Return": cumulative_pct,
                "Annualized Return (%)": annualized_return,
                "Directional Accuracy (%)": directional_accuracy,
                "Number of Trades": len(model_df)
            }
        per_stock_summary[ticker] = ticker_metrics

    print("\n--- Per-Stock Summary by Model ---")
    for ticker, metrics in per_stock_summary.items():
        print(f"\nTicker: {ticker}")
        for model, met in metrics.items():
            print(f"  {model.upper()}: {met}")
    
    # ---------------------------
    # Visualizations (First 2 Graphs)
    # ---------------------------
    # (A) Cumulative Profit Over Time for Each Model.
    comp_df_sorted = comp_df.sort_values(by='next_trading_date')
    plt.figure(figsize=(10, 6))
    for model in ['lstm', 'xgboost', 'rl']:
        profit_col = f"{model}_profit"
        cum_profit = comp_df_sorted[profit_col].cumsum()
        plt.plot(comp_df_sorted['next_trading_date'], cum_profit, marker='o', label=model.upper())
    plt.xlabel("Next Trading Date")
    plt.ylabel("Cumulative Profit ($)")
    plt.title("Cumulative Profit Over Time by Model")
    plt.legend()
    plt.tight_layout()
    plt.show()
    
    # (B) Bar Chart: Average Trade Percentage Return per Model
    avg_lstm_pct = comp_df['lstm_pct'].mean()
    avg_xgboost_pct = comp_df['xgboost_pct'].mean()
    avg_rl_pct = comp_df.loc[comp_df['rl_signal'] != "Hold", 'rl_pct'].mean()
    plt.figure(figsize=(8, 6))
    models_names = ['LSTM', 'XGBOOST', 'RL']
    avg_pcts = [avg_lstm_pct, avg_xgboost_pct, avg_rl_pct]
    plt.bar(models_names, avg_pcts, color=['skyblue', 'salmon', 'lightgreen'])
    plt.ylabel("Average Trade Return (%)")
    plt.title("Average Percentage Return per Trade by Model")
    plt.tight_layout()
    plt.show()
    
    # ---------------------------
    # Return DataFrame and Summaries
    # ---------------------------
    return comp_df, portfolio_summary, overall_annualized, per_stock_summary

