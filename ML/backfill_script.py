import datetime
from datetime import date, timedelta
import pandas as pd
import firebase_admin
from firebase_admin import credentials, firestore

# your existing imports for predict_* and data fetching
from data_processing import fetch_stock_data_alpha, calculate_technical_indicators, preprocess_data_alpha
from run_all_model import (
    get_next_trading_day,
    predict_lstm,
    predict_xgboost,
    predict_nlp,
    predict_rl
)

# initialize Firestore
try:
    firebase_admin.get_app()
except ValueError:
    cred = credentials.Certificate("firebase.json")
    firebase_admin.initialize_app(cred)

db = firestore.client()

def backfill_predictions(tickers, start_date, end_date):
    current = start_date
    while current < end_date:
        # skip weekends
        if current.weekday() < 5:
            pred_date = get_next_trading_day(current)
            for ticker, params in tickers.items():
                # check if prediction already exists
                col = db \
                  .collection("StockSignals") \
                  .document(ticker) \
                  .collection("predictions")
                existing = col \
                  .where("next_trading_date", "==", pred_date.isoformat()) \
                  .get()
                if not existing:
                    print(f"[{current}] Missing {ticker} → {pred_date}, computing...")
                    # 1) Fetch data up to 'current' only:
                    df = fetch_stock_data_alpha(
                        ticker,
                        api_key=params["api_key"],
                        start_date=str(start_date - timedelta(days=365)),
                        end_date=str(current)
                    )
                    df.index = pd.to_datetime(df.index)
                    df['Date']  = df.index.date
                    df = df[df['Date'] <= current].copy()

                    # 2) Generate each model’s prediction:
                    lstm_price = predict_lstm(
                        ticker,
                        params["lstm_model_path"],
                        df
                    )
                    xgb_signal, xgb_prob, _ = predict_xgboost(
                        df,
                        params["xgb_model_path"],
                        params["feature_subset"]
                    )
                    nlp_score, nlp_signal, _ = predict_nlp(
                        ticker,
                        window=params.get("nlp_window", 15),
                        days=params.get("nlp_days", 30)
                    )
                    rl_signal, _ = predict_rl(
                        df,
                        params["dqn_model_path"],
                        params["xgb_model_path"],
                        params["feature_subset"]
                    )

                    # 3) Prepare Firestore document
                    doc = {
                        "lstm_predicted_price": float(lstm_price),
                        "xgboost_signal": int(xgb_signal),
                        "xgboost_prob": float(xgb_prob),
                        "nlp_sentiment_score": float(nlp_score),
                        "nlp_signal": nlp_signal,
                        "rl_recommendation": rl_signal,
                        "next_trading_date": pred_date.isoformat(),
                        # Firestore native timestamp
                        "updated_at": firestore.SERVER_TIMESTAMP
                    }
                    # use e.g. YYYYMMDD as doc ID
                    doc_id = current.strftime("%Y%m%d")
                    col.document(doc_id).set(doc)
                    print(f"  → Backfilled {ticker}/{doc_id}")
                else:
                    print(f"[{current}] {ticker} → {pred_date} already exists.")
        current += timedelta(days=1)

if __name__ == "__main__":
    # Define your tickers & parameters exactly as in your original script:
    tickers = {
        "TSLA": {
            "api_key": "NL9PDOM5JWRPAT9O",
            "lstm_model_path": "models/lstm/lstm_TSLA_model.h5",
            "xgb_model_path": "models/boost/xgboost_TSLA_new.joblib",
            "dqn_model_path": "models/rl/tsla_dqn.h5",
            "feature_subset": ['HT_DCPERIOD', 'WCLPRICE', 'TEMA', 'TRANGE', 'AROON_UP', 'AROONOSC', 'OBV', 'MINUS_DM'],
            "nlp_window": 15,
            "nlp_days": 30
        },
        "MSFT": {
            "api_key": "NL9PDOM5JWRPAT9O",
            "lstm_model_path": "models/lstm/lstm_MSFT_model.h5",
            "xgb_model_path": "models/boost/xgboost_MSFT_new.joblib",
            "dqn_model_path": "models/rl/msft_dqn.h5",
            "feature_subset": ['ADX', 'AD', 'T3', 'BB_upper', 'MFI', 'PLUS_DM', 'RSI', 'AVGPRICE', 'AROON_DOWN', 'MINUS_DI', 'APO', 'BB_lower', 'ADOSC', 'MIDPOINT', 'ROC', 'PLUS_DI', 'MEDPRICE', 'STOCH_fastk', 'NATR'],
            "nlp_window": 15,
            "nlp_days": 30
        },
        "KO": {
            "api_key": "NL9PDOM5JWRPAT9O",
            "lstm_model_path": "models/lstm/lstm_KO_model.h5",
            "xgb_model_path": "models/boost/xgboost_KO_new.joblib",
            "dqn_model_path": "models/rl/ko_dqn.h5",
            "feature_subset": ['ADXR', 'HT_SINE', 'WCLPRICE', 'CMO', 'MFI', 'ATR', 'PLUS_DM', 'RSI', 'AVGPRICE', 'MINUS_DM', 'AROON_DOWN', 'APO', 'BB_lower', 'ADOSC', 'HT_DCPHASE', 'MACD', 'PLUS_DI', 'HT_PHASOR_quadrature', 'MACD_hist', 'TEMA', 'TYPPRICE', 'DEMA', 'MIDPRICE', 'TRANGE', 'NATR'],
            "nlp_window": 15,
            "nlp_days": 30
        }
    }

    start = date(2025, 4, 2)
    tomorrow = date.today() + timedelta(days=1)
    backfill_predictions(tickers, start, tomorrow)
